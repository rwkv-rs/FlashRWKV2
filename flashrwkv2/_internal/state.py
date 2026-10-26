# SPDX-License-Identifier: MIT
"""State-pool transfer shared by packed TMix, CMix and recurrence."""

import torch
import tilelang
import tilelang.language as T


def check_cuda_condition(condition, message):
    """Validate device metadata without host reads in compiled/captured calls."""
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise ValueError(message)


@tilelang.jit(pass_configs={"tl.enable_fast_math": False})
def _transfer(channels: int, batch: int, dtype: str, write: bool):
    slots = T.dynamic("slots")

    @T.prim_func
    def kernel(
        Pool: T.Tensor((slots, channels), dtype),
        Packed: T.Tensor((batch, channels), dtype),
        Indices: T.Tensor((batch,), "int32"),
        Status: T.Tensor((3,), "int32"),
    ):
        with T.Kernel(T.ceildiv(channels, 256), batch, threads=128) as (block, sequence):
            for c in T.Parallel(256):
                channel = block * 256 + c
                if channel < channels:
                    if Status[0] == 0 and sequence < Status[2]:
                        if write:
                            Pool[Indices[sequence], channel] = Packed[sequence, channel]
                        else:
                            Packed[sequence, channel] = Pool[Indices[sequence], channel]
                    elif not write:
                        Packed[sequence, channel] = 0

    return kernel


def gather_state(pool: torch.Tensor, indices: torch.Tensor, status: torch.Tensor) -> torch.Tensor:
    """Gather active slots; inactive sequences receive zero without indexing."""
    packed = pool.new_empty((indices.numel(), *pool.shape[1:]))
    flat = pool.view(pool.shape[0], -1)
    _transfer(flat.shape[1], indices.numel(), str(pool.dtype).split(".")[-1], False)(
        flat, packed.view(indices.numel(), -1), indices, status
    )
    return packed


def scatter_state(
    pool: torch.Tensor, packed: torch.Tensor, indices: torch.Tensor, status: torch.Tensor,
) -> None:
    """Update active slots, preserving the rest of the pool during graph replay."""
    flat = pool.view(pool.shape[0], -1)
    _transfer(flat.shape[1], indices.numel(), str(pool.dtype).split(".")[-1], True)(
        flat, packed.view(indices.numel(), -1), indices, status
    )
