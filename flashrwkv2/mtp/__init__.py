# SPDX-License-Identifier: MIT
"""Single-request LightMTP drafting with reusable CUDA Graph inputs and outputs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .._internal.linear import linear
from ..head.linear import infer_head_linear_all_forward_varlen

_NORMS = ("norm_e", "norm_h", "norm_mlp", "norm_out")


class DraftLoop(nn.Module):
    """Run a shared LightMTP block and an FP32 LM head for each draft token.

    The embedding callable accepts CUDA int64 `[1]` and produces CUDA BF16
    `[1,C]` on the loop's device. Each call returns the same `[1,depth]` output
    buffer. Clone it when results must survive another call on this instance.
    """

    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
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
        gate = state.get("block.gate.weight")
        if not isinstance(gate, torch.Tensor) or gate.ndim != 2 or gate.shape[0] == 0:
            raise ValueError("block.gate.weight must be a nonempty [F,C] tensor")
        features = gate.shape[0]
        shapes = {
            "proj.weight": (channels, 2 * channels),
            "gate.weight": (features, channels),
            "up.weight": (features, channels),
            "down.weight": (channels, features),
        }
        for name in _NORMS:
            shapes[f"{name}.weight"] = shapes[f"{name}.bias"] = (channels,)

        weights = {}
        for name, shape in shapes.items():
            key = f"block.{name}"
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
        self.hidden_size = channels
        self.depth = depth
        self.eps = float(eps)
        self._buffers_for_draft: tuple[torch.Tensor, ...] | None = None
        self._captured: tuple[torch.cuda.CUDAGraph, Callable] | None = None

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
        lm_head_weight: torch.Tensor,
        *,
        depth: int,
        eps: float = 1e-5,
    ) -> DraftLoop:
        """Build from the sidecar's `block.*` weight mapping."""
        return cls(state, lm_head_weight, depth=depth, eps=eps)

    def _apply(self, fn, recurse=True):
        self._captured = None
        self._buffers_for_draft = None
        return super()._apply(fn, recurse=recurse)

    def _workspace(self):
        if self._buffers_for_draft is None:
            if not self.lm_head.is_cuda or self.lm_head.dtype != torch.bfloat16:
                raise ValueError("DraftLoop weights must be CUDA bfloat16 before execution")
            self._buffers_for_draft = (
                self.lm_head.new_zeros((1, self.hidden_size)),
                self.lm_head.new_zeros((1,), dtype=torch.int64),
                self.lm_head.new_empty((1, 2 * self.hidden_size)),
                self.lm_head.new_empty((1, self.depth), dtype=torch.int64),
            )
        return self._buffers_for_draft

    def _load_inputs(self, hidden, token_ids, workspace):
        device = self.lm_head.device
        if (hidden.shape not in ((self.hidden_size,), (1, self.hidden_size))
                or hidden.device != device or hidden.dtype != torch.bfloat16):
            raise ValueError("hidden must be CUDA bfloat16 [1,C] or [C] on the loop device")
        if token_ids.shape not in ((), (1,)) or token_ids.device != device or token_ids.dtype != torch.int64:
            raise ValueError("token_ids must be a CUDA int64 scalar or [1] on the loop device")
        workspace[0].copy_(hidden.reshape(1, self.hidden_size))
        workspace[1].copy_(token_ids.reshape(1))

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
            logits = infer_head_linear_all_forward_varlen(normalize(hidden, "norm_out"), self.lm_head)
            torch.argmax(logits, dim=-1, out=token)
            drafts[:, step].copy_(token)
        return drafts

    @torch.no_grad()
    def forward(
        self, hidden: torch.Tensor, token_ids: torch.Tensor, embed: Callable,
    ) -> torch.Tensor:
        """Run the draft loop eagerly."""
        workspace = self._workspace()
        self._load_inputs(hidden, token_ids, workspace)
        return self._draft(embed, workspace)

    @torch.no_grad()
    def capture(self, embed: Callable) -> None:
        """Warm and capture the draft loop for this embedding and device."""
        if self._captured is not None:
            if embed is not self._captured[1]:
                raise ValueError("this DraftLoop is already captured with a different embedding")
            return
        workspace = self._workspace()
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
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                self._draft(embed, workspace)
        self._captured = (graph, embed)

    @torch.no_grad()
    def replay(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Replay on the current stream using staged device inputs."""
        if self._captured is None:
            raise RuntimeError("capture() must run before replay()")
        workspace = self._workspace()
        self._load_inputs(hidden, token_ids, workspace)
        self._captured[0].replay()
        return workspace[3]


__all__ = ["DraftLoop"]
