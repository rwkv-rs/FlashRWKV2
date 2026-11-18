# SPDX-License-Identifier: MIT

"""CPU compiler and tuning coverage for functional and indexed recurrence."""

import pytest
import torch


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("fault", ["output", "pool"])
def test_recurrent_tuning_rejects_indexed_only_errors(monkeypatch, indexed, fault):
    import _tuning as tuning
    from flashrwkv2.tmix.wkv7 import tilelang as implementation

    values = [torch.ones(3, 1, 64, dtype=torch.bfloat16) for _ in range(6)]
    initial = torch.zeros(2, 1, 64, 64)
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    calls = []

    def functional(*args, block_values, **kwargs):
        calls.append("functional")
        return torch.ones_like(args[0]), args[6] + 1

    def inference(*args, block_values, **kwargs):
        calls.append("indexed")
        pool, indices, status = args[6], args[8], args[9]
        assert pool.data_ptr() != initial.data_ptr()
        assert indices.tolist() == [0, 1]
        assert status.tolist() == [0, 3, 2]
        output = torch.ones_like(args[0])
        pool.add_(1)
        if kwargs["prefetch_tokens"] == 4:
            if fault == "output":
                output.add_(1)
            else:
                pool[0, 0, 0, 0] += 1e-5  # Preserve an FP32-only difference.
        return output

    def select_config(candidates, run, *, check, reference):
        assert len(reference) == 4 and reference[3].dtype == torch.float32
        measurements = []
        for config in candidates:
            actual = check(config)
            equal = all(torch.equal(a, b) for a, b in zip(actual, reference, strict=True))
            measurements.append({"config": config, "bitwise_equal": equal})
        assert [item["config"] for item in measurements if item["bitwise_equal"]] == [
            (64, 1), (32, 1),
        ]
        run((64, 1))
        assert calls[-1] == ("indexed" if indexed else "functional")
        return (64, 1), measurements

    monkeypatch.setattr(implementation, "tmix_wkv7_recurrent", functional)
    monkeypatch.setattr(implementation, "_infer_recurrent", inference)
    monkeypatch.setattr(tuning, "select_config", select_config)
    result = tuning.tune_tmix_wkv7_recurrent(*values, initial, cu, indexed=indexed)
    assert result["block_values"] == 64 and result["prefetch_tokens"] == 1
    assert result["indexed"] == indexed
    assert torch.count_nonzero(initial) == 0
