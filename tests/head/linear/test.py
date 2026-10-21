# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch

from flashrwkv2.head.linear import (
    head_logprobs,
    infer_head_linear_all_forward_varlen,
    infer_head_linear_last_forward_varlen,
    pretrain_head_linear_bf16,
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("vocab", (17, 1025, 65536))
def test_selected_logprobs_fixed_reduction_and_vjp(vocab):
    torch.manual_seed(124)
    logits = torch.randn(7, vocab, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 3, 5, vocab - 1, 1, 3, 0], device="cuda")
    output = head_logprobs(logits, targets)
    reference = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    torch.testing.assert_close(output, reference, atol=4e-6, rtol=2e-6)
    separate = torch.cat([
        head_logprobs(logits[row:row + 1], targets[row:row + 1]) for row in range(7)
    ])
    assert torch.equal(output, separate)
    dy = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, logits, dy, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference, logits, dy)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("offset", (0.0, 1e8))
def test_selected_logprobs_masked_vocabulary_and_large_offset(offset):
    logits = torch.full((2, 2048), offset, device="cuda", requires_grad=True)
    with torch.no_grad():
        logits[:, 1024:] = -torch.inf
    targets = torch.tensor([0, 1013], device="cuda")
    actual = head_logprobs(logits, targets)
    expected = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    gradients = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    references = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(gradients, references, atol=1e-7, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.memcheck
@pytest.mark.parametrize(("rows", "channels", "vocab"), ((4, 8, 11), (1, 4096, 65536)))
def test_head_all_and_last_linear(rows: int, channels: int, vocab: int) -> None:
    torch.manual_seed(59)
    device = torch.device("cuda")
    x = torch.randn(rows, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(vocab, channels, device=device, dtype=torch.bfloat16)
    all_logits = infer_head_linear_all_forward_varlen(x, weight)
    assert all_logits.dtype == torch.float32
    assert torch.allclose(
        all_logits.float(), x.float() @ weight.float().t(), atol=0.04, rtol=0.04
    )
    x.requires_grad_()
    for repeat in range(4):
        outputs = (
            ("training", pretrain_head_linear_bf16(x, weight)),
            ("all", infer_head_linear_all_forward_varlen(x.detach(), weight)),
            ("last", infer_head_linear_last_forward_varlen(
                x.detach(), weight, tokens_count=max(rows, 2)
            )),
        )
        for name, actual in outputs:
            assert torch.equal(actual, all_logits), (
                f"{name} repeat {repeat}: "
                f"{torch.count_nonzero(actual != all_logits).item()} unequal logits; "
                f"first indices={torch.nonzero(actual != all_logits)[:8].tolist()}"
            )
    with pytest.raises(ValueError, match="positive"):
        infer_head_linear_last_forward_varlen(x, weight, tokens_count=0)


def test_head_linear_rejects_cpu_before_native_launch() -> None:
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    weight = torch.zeros(11, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        infer_head_linear_all_forward_varlen(x, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("use_nora", "adapter_dtype"), [(True, torch.float32), (False, torch.bfloat16)]
)
def test_head_adapter_training_and_merge(use_nora, adapter_dtype):
    from utils import assert_relative_rmse

    torch.manual_seed(71)
    x = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16) * 0.03
    a = (
        torch.randn(256, 64, device="cuda", dtype=adapter_dtype) * 0.05
    ).requires_grad_()
    b = (
        torch.randn(257, 256, device="cuda", dtype=adapter_dtype) * 0.01
    ).requires_grad_()
    refs = [tensor.detach().clone().requires_grad_() for tensor in (x, a, b)]
    output = pretrain_head_linear_bf16(
        x, weight, lora_a=a, lora_b=b, lora_scale=0.7, use_nora=use_nora
    )
    with torch.no_grad():
        inference = infer_head_linear_all_forward_varlen(
            x, weight, lora_a=a, lora_b=b, lora_scale=0.7, use_nora=use_nora
        )
    assert torch.equal(output, inference)
    normalized = refs[1].float()
    if use_nora:
        normalized = normalized / (normalized.norm(dim=0, keepdim=True) + 1.0e-6)
    features = (refs[0].float() @ normalized.to(torch.bfloat16).float().t()).to(
        torch.bfloat16
    )
    expected = refs[0].float() @ weight.float().t()
    expected = expected + 0.7 * (
        features.float() @ refs[2].to(torch.bfloat16).float().t()
    )
    assert output.dtype == torch.float32
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (x, a, b), upstream)
    expected_grads = torch.autograd.grad(expected, refs, upstream)
    assert_relative_rmse(output, expected)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        assert_relative_rmse(actual, reference)
    with torch.no_grad():
        a.add_(actual_grads[1], alpha=-0.001)
        b.add_(actual_grads[2], alpha=-0.001)
    updated = pretrain_head_linear_bf16(
        x, weight, lora_a=a, lora_b=b, lora_scale=0.7, use_nora=use_nora
    )
    with torch.no_grad():
        inference = infer_head_linear_last_forward_varlen(
            x, weight, tokens_count=1,
            lora_a=a, lora_b=b, lora_scale=0.7, use_nora=use_nora,
        )
    assert torch.equal(updated, inference)
    normalized = a.float()
    if use_nora:
        normalized = normalized / (normalized.norm(dim=0, keepdim=True) + 1.0e-6)
    merged = weight.float() + 0.7 * b.float() @ normalized
    assert_relative_rmse(updated, x.float() @ merged.t())
    assert not torch.equal(output, updated)
    assert weight.grad is None


class _FSDPNoRAProjection(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(257, 64, device=device) * 0.03, requires_grad=False
        )
        self.a = torch.nn.Parameter(torch.randn(256, 64, device=device) * 0.03)
        self.b = torch.nn.Parameter(torch.randn(257, 256, device=device) * 0.01)

    def _project(self, x):
        from flashrwkv2 import pretrain_head_linear_bf16

        return pretrain_head_linear_bf16(
            x,
            self.weight.to(x.dtype),
            lora_a=self.a.to(x.dtype),
            lora_b=self.b.to(x.dtype),
            use_nora=True,
        )

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._project, x, use_reentrant=False)


def _fsdp_nora_training_worker(rank, store_path):
    import copy
    import io

    import torch.distributed as dist
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from utils import assert_relative_rmse

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{store_path}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(97)
        model = _FSDPNoRAProjection(torch.device("cuda", rank))
        reference = copy.deepcopy(model)
        initial_a = model.a.detach().clone()
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(101 + rank)
        x = torch.randn(5, 64, device=torch.device("cuda", rank), dtype=torch.bfloat16)
        target = torch.randn(5, 257, device=torch.device("cuda", rank))
        loss = (model(x).float() - target).square().mean()
        loss.backward()
        reference_loss = (reference(x).float() - target).square().mean()
        reference_loss.backward()
        for name in ("a", "b"):
            expected = getattr(reference, name).grad
            dist.all_reduce(expected)
            expected.div_(2)
            actual = getattr(model, name).grad.full_tensor()
            assert_relative_rmse(actual, expected)
        assert model.weight.grad is None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        trained = model(x).detach()
        state = {
            name: value.full_tensor().detach()
            for name, value in model.state_dict().items()
        }
        assert not torch.equal(initial_a, state["a"])
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = _FSDPNoRAProjection(torch.device("cuda", rank))
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        assert_relative_rmse(trained, restored(x), 0.001)
        normalized = state["a"] / (state["a"].norm(dim=0, keepdim=True) + 1e-6)
        merged = state["weight"] + state["b"] @ normalized
        assert_relative_rmse(trained, x.float() @ merged.t())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two CUDA devices are required"
)
@pytest.mark.cuda
def test_fsdp_nora_checkpoint_step_and_merge(tmp_path):
    torch.multiprocessing.spawn(
        _fsdp_nora_training_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True
    )
