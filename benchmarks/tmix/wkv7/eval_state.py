# SPDX-License-Identifier: MIT
"""Score WKV inference state precision with a local 7.2B HF checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def score_lambada(model, tokenizer, path, batch_size):
    samples = []
    for index, line in enumerate(path.read_text().splitlines()):
        prefix, word = json.loads(line)["text"].rsplit(" ", 1)
        prompt = [0, *tokenizer.encode(prefix, add_special_tokens=False)]
        target = tokenizer.encode(" " + word, add_special_tokens=False)
        samples.append((len(prompt) + len(target), index, prompt, target))
    samples.sort()
    per_word = [None] * len(samples)
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        length = max(len(p) + len(t) - 1 for _, _, p, t in batch)
        words = max(len(t) for _, _, _, t in batch)
        inputs, positions, labels = [], [], []
        for _, _, prompt, target in batch:
            ids = prompt + target[:-1]
            inputs.append(ids + [0] * (length - len(ids)))
            positions.append([len(prompt) - 1 + i for i in range(len(target))] + [0] * (words - len(target)))
            labels.append(target + [-100] * (words - len(target)))
        ids, selected, targets = (torch.tensor(t, device="cuda") for t in (inputs, positions, labels))
        logits = model(ids, logits_to_keep=selected, use_cache=False).logits
        nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        totals = nll.double().sum(1).tolist()
        correct = ((logits.argmax(-1) == targets) | (targets == -100)).all(1).tolist()
        for (_, index, _, target), value, hit in zip(batch, totals, correct, strict=True):
            per_word[index] = (value, len(target), hit)
    nll = sum(value for value, _, _ in per_word)
    count = len(per_word)
    return {
        "words": count, "word_ppl": math.exp(nll / count),
        "target_token_ppl": math.exp(nll / sum(n for _, n, _ in per_word)),
        "accuracy": sum(hit for _, _, hit in per_word) / count, "per_word": per_word,
    }


def score_long_context(model, tokenizer, paths, chunk_size):
    from transformers.models.rwkv.modeling_rwkv import RwkvCache

    bins = [0.0] * 4
    for path in paths:
        text = path.read_text()
        ids = [0, *tokenizer.encode(text, add_special_tokens=False)[:32768]]
        if len(ids) < 32769:
            raise ValueError(f"{path} needs at least 32768 text tokens")
        ids = torch.tensor(ids, device="cuda")
        cache = RwkvCache(model.config)
        for bucket in range(4):
            begin, end = bucket * 8192, (bucket + 1) * 8192
            subtotal = torch.zeros((), dtype=torch.float64, device="cuda")
            for start in range(begin, end, chunk_size):
                stop = min(start + chunk_size, end)
                logits = model(ids[None, start:stop], past_key_values=cache, use_cache=True).logits[0]
                losses = F.cross_entropy(logits, ids[start + 1:stop + 1], reduction="none")
                subtotal += losses.double().sum()
            bins[bucket] += subtotal.item()
    count = 8192 * len(paths)
    return [{"target_positions": [i * 8192 + 1, (i + 1) * 8192], "tokens": count,
             "ppl": math.exp(total / count)} for i, total in enumerate(bins)]


@torch.inference_mode()
def main():
    from transformers import AutoTokenizer, RwkvForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--state-dtype", required=True, choices=("float16", "float32"))
    parser.add_argument("--lambada", required=True, type=Path)
    parser.add_argument("--long-texts", required=True, nargs="+", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = RwkvForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model.config.wkv_mode = "fp16" if args.state_dtype == "float16" else "fp32io16"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    result = {
        "model": args.model.name, "state_dtype": args.state_dtype,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
        "batch_size": args.batch_size, "chunk_size": args.chunk_size,
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (args.lambada, *args.long_texts)},
        "lambada": score_lambada(model, tokenizer, args.lambada, args.batch_size),
    }
    args.output.write_text(json.dumps(result) + "\n")
    result["long_context"] = score_long_context(model, tokenizer, args.long_texts, args.chunk_size)
    args.output.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
