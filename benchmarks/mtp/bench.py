# SPDX-License-Identifier: MIT
"""Compare the LightMTP draft loop with a PyTorch reference at the same shape."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from _timing import measure_cuda

from flashrwkv2.mtp import DraftLoop


def _weights(channels, features, vocab):
    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    block = {
        "block.proj.weight": randn(channels, 2 * channels) * 0.02,
        "block.gate.weight": randn(features, channels) * 0.02,
        "block.up.weight": randn(features, channels) * 0.02,
        "block.down.weight": randn(channels, features) * 0.02,
    }
    for name in ("norm_e", "norm_h", "norm_mlp", "norm_out"):
        block[f"block.{name}.weight"] = randn(channels) * 0.1 + 1
        block[f"block.{name}.bias"] = torch.zeros_like(block[f"block.{name}.weight"])
    return block, randn(vocab, channels).mul_(0.02), randn(vocab, channels)


def _torch_draft(block, head, embed, depth, hidden, token):
    def norm(x, name):
        return F.layer_norm(x, (x.shape[-1],), block[f"block.{name}.weight"], block[f"block.{name}.bias"])

    drafts = []
    for _ in range(depth):
        pair = torch.cat([norm(embed(token), "norm_e"), norm(hidden, "norm_h")], dim=-1)
        x = hidden + F.linear(pair, block["block.proj.weight"])
        mlp = norm(x, "norm_mlp")
        gate = F.silu(F.linear(mlp, block["block.gate.weight"]))
        hidden = x + F.linear(gate * F.linear(mlp, block["block.up.weight"]), block["block.down.weight"])
        logits = torch.mm(norm(hidden, "norm_out"), head.T, out_dtype=torch.float32)
        token = logits.argmax(dim=-1)
        del logits
        drafts.append(token)
    return torch.stack(drafts, dim=1)


def _capture_reference(draft, hidden, token):
    static_hidden, static_token = hidden.clone(), token.clone()
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        draft(static_hidden, static_token)
    current.wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        result = draft(static_hidden, static_token)

    def replay():
        static_hidden.copy_(hidden)
        static_token.copy_(token)
        graph.replay()
        return result

    return replay


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 64, 320, 512])
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    block, head, embedding = _weights(args.hidden, args.intermediate, args.vocab)
    loop = DraftLoop.from_state_dict(block, head, depth=args.depth)

    def embed(ids):
        return F.embedding(ids, embedding)

    def torch_draft(h, ids):
        return _torch_draft(block, head, embed, args.depth, h, ids)

    profiles = []
    torch_reference_mismatches = {}
    for batch_size in args.batch_sizes:
        hidden = torch.randn(batch_size, args.hidden, device="cuda", dtype=torch.bfloat16)
        token = torch.randint(args.vocab, (batch_size,), device="cuda")
        tilelang_expected = loop(hidden, token, embed).clone()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loop.capture(embed, batch_size=batch_size)
        capture_peak = torch.cuda.max_memory_allocated() - baseline
        variants = {
            "torch_eager": lambda: torch_draft(hidden, token),
            "torch_graph": _capture_reference(torch_draft, hidden, token),
            "tilelang_eager": lambda: loop(hidden, token, embed),
            "tilelang_graph": lambda: loop.replay(hidden, token),
        }
        torch_expected = torch_draft(hidden, token)
        torch_reference_mismatches[batch_size] = (torch_expected != tilelang_expected).count_nonzero().item()
        for name, run in variants.items():
            expected = tilelang_expected if name.startswith("tilelang_") else torch_expected
            if not torch.equal(run(), expected):
                raise AssertionError(f"{name} draft IDs differ at batch_size={batch_size}")

        for name, run in variants.items():
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            timing = measure_cuda(run, warmup=10, batches=10, launches_per_batch=40)
            profiles.append({
                "profile": f"b{batch_size}/d{args.depth}/{name}",
                "batch_size": batch_size,
                "allocated_bytes_before_timing": allocated,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "capture_peak_extra_bytes": capture_peak if name == "tilelang_graph" else None,
                **timing,
            })
    print(json.dumps({
        "profiles": profiles,
        "shape": [args.hidden, args.intermediate, args.vocab],
        "depth": args.depth,
        "seed": args.seed,
        "head_read_bytes_per_draft_loop": args.depth * head.nbytes,
        "torch_reference_mismatched_draft_ids": torch_reference_mismatches,
    }))


if __name__ == "__main__":
    main()
