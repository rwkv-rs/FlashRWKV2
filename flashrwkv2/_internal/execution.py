# SPDX-License-Identifier: MIT
"""Device identity, workload buckets and native compilation cache location."""

import os
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch


class _AutogradContext(SimpleNamespace):
    """Reuse an owner's eager forward/VJP inside an opaque compiled operator."""

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def set_materialize_grads(self, value):
        """The registered autograd context controls gradient materialization."""


@cache
def native_cache_root():
    """Return the root used by the native extension compiler."""
    configured = os.environ.get("FLASHRWKV2_CACHE_DIR") or "~/.cache/flashrwkv2"
    return Path(configured).expanduser().resolve()


@cache
def device_class(device):
    props = torch.cuda.get_device_properties(device)
    return (props.name, props.major, props.minor, props.multi_processor_count)


def _fixed_strategy_device(info):
    return info[1:] in ((8, 9, 128), (12, 0, 188))


def workload_bucket(size):
    """At most eleven workload classes; model dimensions remain exact."""
    return 0 if size == 0 else min(512, 1 << (size - 1).bit_length())
