# SPDX-License-Identifier: MIT
"""Batched LightMTP drafting with reusable CUDA Graph inputs and outputs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .._internal.linear import check_adapter_parameters, check_adapter_scale, linear
from ..head.linear import infer_head_linear_all_forward_varlen

_NORMS = ("norm_e", "norm_h", "norm_mlp", "norm_out")


class DraftLoop(nn.Module):
    """Run a shared LightMTP block and an FP32 LM head for each draft token.

    The embedding callable accepts CUDA int64 `[B]` and produces CUDA BF16
    `[B,C]` on the loop's device. Calls of the same batch size share a
    `[B,depth]` output buffer. Clone it to retain the IDs across those calls.
    """

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
        lora_a: torch.Tensor | None = None,
        lora_b: torch.Tensor | None = None,
        lora_scale: float = 1.0,
        use_nora: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if (
            not isinstance(lm_head_weight, torch.Tensor) or lm_head_weight.ndim != 2
            or lm_head_weight.numel() == 0 or not lm_head_weight.is_floating_point()
        ):
            raise ValueError("lm_head_weight must be a nonempty floating-point [V,C] tensor")

        channels = lm_head_weight.shape[1]
        device = lm_head_weight.device
        norm_keys = tuple(f"{name}.{param}" for name in _NORMS for param in ("weight", "bias"))
        required = ("proj.weight", "gate.weight", "up.weight", "down.weight", *norm_keys)
        prefix = "block"
        if not all(f"block.{name}" in state for name in required) and any(f"mtp.{name}" in state for name in required):
            prefix = "mtp"
        gate_key = f"{prefix}.gate.weight"
        gate = state.get(gate_key)
        if not isinstance(gate, torch.Tensor) or gate.ndim != 2 or gate.shape[0] == 0:
            raise ValueError(f"{gate_key} must be a nonempty [F,C] tensor")
        features = gate.shape[0]
        shapes = {
            "proj.weight": (channels, 2 * channels),
            "gate.weight": (features, channels),
            "up.weight": (features, channels),
            "down.weight": (channels, features),
        }
        shapes.update(dict.fromkeys(norm_keys, (channels,)))

        weights = {}
        for name, shape in shapes.items():
            key = f"{prefix}.{name}"
            weight = state.get(key)
            if (
                not isinstance(weight, torch.Tensor) or weight.shape != shape
                or not weight.is_floating_point()
            ):
                raise ValueError(f"{key} must be a floating-point tensor with shape {shape}")
            weights[name.replace(".", "_")] = nn.Parameter(
                weight.detach().to(device=device, dtype=torch.bfloat16).contiguous(),
                requires_grad=False,
            )
        self._weights = nn.ParameterDict(weights)
        self.lm_head = nn.Parameter(
            lm_head_weight.detach().to(dtype=torch.bfloat16).contiguous(), requires_grad=False,
        )
        if (lora_a is not None and not isinstance(lora_a, torch.Tensor)) or (
            lora_b is not None and not isinstance(lora_b, torch.Tensor)
        ):
            raise TypeError("lora_a and lora_b must be torch.Tensor or None")
        lora_a = lora_a.detach().to(device=device).contiguous() if lora_a is not None else None
        lora_b = lora_b.detach().to(device=device).contiguous() if lora_b is not None else None
        check_adapter_parameters(lora_a, lora_b, lora_scale, self.lm_head, device)
        self.register_buffer("lora_a", lora_a)
        self.register_buffer("lora_b", lora_b)
        self.lora_scale = check_adapter_scale(lora_scale)
        self.use_nora = bool(use_nora)
        self.hidden_size = channels
        self.depth = depth
        self.eps = float(eps)
        self._buffers_for_draft: dict[int, tuple[torch.Tensor, ...]] = {}
        self._captured: dict[int, tuple[torch.cuda.CUDAGraph, Callable]] = {}
        self._graph_pool = None

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
        lora_a: torch.Tensor | None = None,
        lora_b: torch.Tensor | None = None,
        lora_scale: float = 1.0,
        use_nora: bool = False,
    ) -> DraftLoop:
        """Load a complete `mtp.*` block or a legacy `block.*` sidecar.

        A complete legacy mapping takes precedence. Required projection and
        norm weights must all share one prefix. LoRA/NoRA options apply to
        the LM head.
        """
        return cls(
            state, lm_head_weight, depth=depth, eps=eps, lora_a=lora_a, lora_b=lora_b,
            lora_scale=lora_scale, use_nora=use_nora,
        )

    def _invalidate(self) -> None:
        self._captured.clear()
        self._buffers_for_draft.clear()
        self._graph_pool = None

    def _apply(self, fn, recurse=True):
        self._invalidate()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate()
        super()._load_from_state_dict(*args, **kwargs)

    def _workspace(self, batch_size):
        if batch_size not in self._buffers_for_draft:
            if not self.lm_head.is_cuda or self.lm_head.dtype != torch.bfloat16:
                raise ValueError("DraftLoop weights must be CUDA bfloat16 before execution")
            self._buffers_for_draft[batch_size] = (
                self.lm_head.new_zeros((batch_size, self.hidden_size)),
                self.lm_head.new_zeros((batch_size,), dtype=torch.int64),
                self.lm_head.new_empty((batch_size, 2 * self.hidden_size)),
                self.lm_head.new_empty((batch_size, self.depth), dtype=torch.int64),
            )
        return self._buffers_for_draft[batch_size]

    def _input_batch_size(self, hidden, token_ids):
        device = self.lm_head.device
        if (
            hidden.ndim not in (1, 2) or hidden.shape[-1] != self.hidden_size or hidden.shape[0] == 0
            or hidden.device != device or hidden.dtype != torch.bfloat16 or not hidden.is_cuda
        ):
            raise ValueError("hidden must be CUDA bfloat16 [B,C] or [C] on the loop device, with B >= 1")
        batch_size = hidden.shape[0] if hidden.ndim == 2 else 1
        if (
            (token_ids.shape != (batch_size,) and not (batch_size == 1 and token_ids.ndim == 0))
            or token_ids.device != device or token_ids.dtype != torch.int64
        ):
            raise ValueError("token_ids must be CUDA int64 [B] on the loop device, or a scalar for B=1")
        return batch_size

    def _draft(self, embed, workspace):
        from ..post_norm.tilelang import norm

        hidden, token, joined, drafts = workspace
        weights = self._weights
        channels = self.hidden_size

        def normalize(x, name):
            return norm(x, weights[f"{name}_weight"], weights[f"{name}_bias"], self.eps)

        for step in range(self.depth):
            joined[:, :channels].copy_(normalize(embed(token), "norm_e"))
            joined[:, channels:].copy_(normalize(hidden, "norm_h"))
            x = hidden + linear(joined, weights["proj_weight"])
            mlp = normalize(x, "norm_mlp")
            gate = linear(mlp, weights["gate_weight"])
            up = linear(mlp, weights["up_weight"])
            hidden = x + linear(F.silu(gate) * up, weights["down_weight"])
            logits = infer_head_linear_all_forward_varlen(
                normalize(hidden, "norm_out"), self.lm_head, lora_a=self.lora_a, lora_b=self.lora_b,
                lora_scale=self.lora_scale, use_nora=self.use_nora,
            )
            torch.argmax(logits, dim=-1, out=token)
            del logits
            drafts[:, step].copy_(token)
        return drafts

    @torch.no_grad()
    def forward(
        self, hidden: torch.Tensor, token_ids: torch.Tensor, embed: Callable,
    ) -> torch.Tensor:
        """Run the draft loop eagerly."""
        workspace = self._workspace(self._input_batch_size(hidden, token_ids))
        workspace[0].copy_(hidden)
        workspace[1].copy_(token_ids)
        return self._draft(embed, workspace)

    @torch.no_grad()
    def capture(self, embed: Callable, batch_size: int = 1) -> None:
        """Warm and capture a graph for this batch size, embedding and device."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size in self._captured:
            if embed is not self._captured[batch_size][1]:
                raise ValueError("this DraftLoop batch size is already captured with a different embedding")
            return
        workspace = self._workspace(batch_size)
        device = self.lm_head.device
        with torch.cuda.device(device):
            current = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                workspace[0].zero_()
                workspace[1].zero_()
                self._draft(embed, workspace)
            current.wait_stream(side)
            if self._graph_pool is None:
                self._graph_pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool, stream=side):
                self._draft(embed, workspace)
        self._captured[batch_size] = (graph, embed)

    @torch.no_grad()
    def replay(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Replay on the current stream using staged device inputs."""
        if not self._captured:
            raise RuntimeError("capture() must run before replay()")
        batch_size = self._input_batch_size(hidden, token_ids)
        if batch_size not in self._captured:
            raise RuntimeError(f"capture() must run for batch_size={batch_size} before replay()")
        workspace = self._workspace(batch_size)
        workspace[0].copy_(hidden)
        workspace[1].copy_(token_ids)
        self._captured[batch_size][0].replay()
        return workspace[3]


__all__ = ["DraftLoop"]
