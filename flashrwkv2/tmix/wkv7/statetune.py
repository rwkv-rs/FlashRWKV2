# SPDX-License-Identifier: MIT

"""State tuning through the same BF16 forward used by inference."""

from itertools import pairwise

import torch


def statetune_tmix_wkv7_recurrent_fp32io16(
    initial_state,
    sequence_chunk_offsets,
    chunk_token_starts,
    chunk_token_ends,
    r,
    decay_logits,
    k,
    v,
    a,
    b,
    *,
    scale: float = 1.0,
):
    """Return ``(output, final_state)`` with gradients to the initial state.

    The chunk records partition packed tokens in sequence order. They locate
    sequence boundaries; the shared recurrence manages backward checkpoints
    internally and accepts arbitrary sequence/chunk lengths.
    """
    for name, tensor in (
        ("sequence_chunk_offsets", sequence_chunk_offsets),
        ("chunk_token_starts", chunk_token_starts),
        ("chunk_token_ends", chunk_token_ends),
    ):
        if (
            tensor.dtype != torch.int32
            or tensor.device != r.device
            or not tensor.is_contiguous()
            or tensor.ndim != 1
        ):
            raise ValueError(f"{name} must be contiguous CUDA int32")
    if sequence_chunk_offsets.shape != (initial_state.shape[0] + 1,):
        raise ValueError("sequence_chunk_offsets must have shape [B+1]")
    if (
        chunk_token_starts.numel() == 0
        or chunk_token_ends.shape != chunk_token_starts.shape
    ):
        raise ValueError("chunk token metadata must have matching nonempty shape [C]")
    metadata = torch.cat(
        (sequence_chunk_offsets, chunk_token_starts, chunk_token_ends)
    ).tolist()
    offsets_count = sequence_chunk_offsets.numel()
    chunks = chunk_token_starts.numel()
    offsets = metadata[:offsets_count]
    starts = metadata[offsets_count : offsets_count + chunks]
    ends = metadata[offsets_count + chunks :]
    if (
        offsets[0] != 0
        or offsets[-1] != chunks
        or any(right < left for left, right in pairwise(offsets))
    ):
        raise ValueError("sequence_chunk_offsets must be nondecreasing and span [0,C]")
    if (
        starts[0] != 0
        or ends[-1] != r.shape[0]
        or starts[1:] != ends[:-1]
        or any(end <= start for start, end in zip(starts, ends, strict=True))
    ):
        raise ValueError("chunk token ranges must form a contiguous partition of [0,N]")
    # The terminal sentinel also handles a sequence with no chunks.
    starts_with_end = [*starts, ends[-1]]
    cu_seqlens = torch.tensor(
        [starts_with_end[offset] for offset in offsets],
        dtype=torch.int32,
        device=r.device,
    )
    from .tilelang import tmix_wkv7_recurrent

    return tmix_wkv7_recurrent(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        initial_state,
        cu_seqlens,
        scale=scale,
    )


__all__ = ["statetune_tmix_wkv7_recurrent_fp32io16"]
