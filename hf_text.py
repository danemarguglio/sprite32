"""Frozen LLM text encoder for cross-attention (Qwen3-4B), with an on-disk cache.

Token states are hidden_states[llm_layer] (default -2) of the causal LM. Qwen has no BOS token and the state at
position 0 is an attention sink (norm ~6,000 against ~600 for the rest, measured 2026-11-11), so every caption is
prefixed with one `<|im_start|>` token whose state is dropped. An empty caption has no tokens at all (mask all 0);
the UNet's TextAdapter prepends a learned, always-visible token.

Encoding live costs ~225 ms per 128 captions on a 5090 against ~120 ms for the UNet step, so training reads a cache
built once by scripts/build_text_cache.py: fp16 states of every caption, keyed by a hash of the caption text.
Captions missing from the cache (prompts typed at sampling time) are encoded live; the LLM is only loaded then.

The pooled vector for the embedding stream stays CLIP's EOS state (hf_common.TextEncoder), as in every run so far.
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hf_common import TextEncoder


def caption_hash(caption):
    return int.from_bytes(hashlib.blake2b(caption.encode("utf-8"), digest_size=8).digest(), "little")


class LLMEncoder:
    """Per-token states of a frozen causal LM, sink position dropped."""

    def __init__(self, name, device, max_tokens=64, layer=-2, prefix="<|im_start|>"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"  # causal: padding after the text cannot change its states
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(torch.device(device).type, torch.float32)
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval().requires_grad_(False)
        self.device, self.max_tokens, self.layer, self.prefix = device, max_tokens, layer, prefix
        self.dim = self.model.config.hidden_size

    def lengths(self, captions):
        """Tokens kept per caption (without the prefix), as __call__ would return them."""
        ids = self.tokenizer([self.prefix + c for c in captions], truncation=True, max_length=self.max_tokens)["input_ids"]
        return [len(i) - 1 for i in ids]

    @torch.no_grad()
    def __call__(self, captions):
        """-> states (B, T, dim) in the model dtype, mask (B, T) bool; T = longest caption in the batch (>= 1)."""
        tok = self.tokenizer([self.prefix + c for c in captions], padding=True, truncation=True,
                             max_length=self.max_tokens, return_tensors="pt").to(self.device)
        h = self.model(**tok, output_hidden_states=True).hidden_states[self.layer]
        h, mask = h[:, 1:], tok.attention_mask[:, 1:].bool()
        if h.shape[1] == 0:  # only empty captions
            h, mask = h.new_zeros(len(captions), 1, self.dim), mask.new_zeros(len(captions), 1)
        return h, mask


class TextCache:
    """Read side of scripts/build_text_cache.py: tokens.f16 (n_tokens, dim) + per-caption (hash, start, len).
    Read with pread rather than a memmap: pages a process has mapped cannot be dropped from the page cache, which
    hf_train.trim_page_cache has to do in a container with a memory limit."""

    def __init__(self, path):
        path = Path(path)
        self.meta = json.load(open(path / "meta.json"))
        index = np.load(path / "index.npz")
        order = np.argsort(index["hash"])
        self.hash, self.start, self.len = index["hash"][order], index["start"][order], index["len"][order]
        self.fd = os.open(path / "tokens.f16", os.O_RDONLY)
        self.row = 2 * self.meta["dim"]  # bytes per token
        self.pool = ThreadPoolExecutor(8)
        stats = np.load(path / "stats.npz")
        self.mean, self.std = torch.from_numpy(stats["mean"]), torch.from_numpy(stats["std"])

    def get(self, caption):
        """(len, dim) fp16 tensor, or None if the caption is not cached."""
        h = np.uint64(caption_hash(caption))
        i = int(np.searchsorted(self.hash, h))
        if i == len(self.hash) or self.hash[i] != h:
            return None
        n = int(self.len[i])
        buf = bytearray(os.pread(self.fd, n * self.row, int(self.start[i]) * self.row))
        return torch.frombuffer(buf, dtype=torch.float16).view(n, -1) if n else torch.zeros(0, self.meta["dim"],
                                                                                           dtype=torch.float16)

    def get_many(self, captions):
        """get() for each caption, on 8 threads: a batch is 128 scattered reads, which one after another take 24 ms
        when the file is not in the page cache (RunPod NVMe, 2026-11-11)."""
        return list(self.pool.map(self.get, captions))


class LLMText:
    """The (ctx, mask, pooled) interface of hf_common.TextEncoder: ctx/mask from the LLM (cache first, live for
    misses), padded to llm_max_tokens - 1; pooled from CLIP. ctx is fp16 on the device; the TextAdapter casts it."""

    def __init__(self, cfg, device):
        self.clip = TextEncoder(cfg["text_model"], device, cfg.get("max_text_tokens", 77))
        self.model = self.clip.model  # what compile_text compiles
        self.cfg, self.device = cfg, device
        self.spec = {"model": cfg["llm_model"], "layer": cfg.get("llm_layer", -2),
                     "max_tokens": cfg.get("llm_max_tokens", 64), "prefix": cfg.get("llm_prefix", "<|im_start|>")}
        self.cache, self.llm, self.memo, self.misses = None, None, {}, 0
        cache_dir = cfg.get("text_cache_dir")
        if cache_dir and (Path(cache_dir).expanduser() / "meta.json").exists():
            self.cache = TextCache(Path(cache_dir).expanduser())
            got = {k: self.cache.meta[k] for k in self.spec}
            if got != self.spec:
                raise ValueError(f"text cache {cache_dir} was built with {got}, the config asks for {self.spec}")
            print(f"text cache {cache_dir}: {len(self.cache.hash)} captions, {self.cache.meta['n_tokens']} tokens")
        else:
            print(f"no text cache at {cache_dir}: encoding every caption with {self.spec['model']} (slow for training)")
        self.dim = self.cache.meta["dim"] if self.cache else self._llm().dim

    def _llm(self):
        if self.llm is None:
            print(f"loading {self.spec['model']} for captions that are not in the text cache", flush=True)
            self.llm = LLMEncoder(self.spec["model"], self.device, self.spec["max_tokens"], self.spec["layer"],
                                  self.spec["prefix"])
        return self.llm

    def release(self):
        """Drop the live LLM (8 GB of GPU memory for Qwen3-4B); it is loaded again on the next cache miss.
        hf_train.py calls this before a benchmark subprocess, which has to fit on the same GPU."""
        if self.llm is not None:
            self.llm = None
            torch.cuda.empty_cache()

    def tokens(self, captions):
        """List of (len_i, dim) fp16 CPU tensors; len 0 for an empty caption."""
        out = [None] * len(captions)
        look = [i for i, c in enumerate(captions) if c and c not in self.memo] if self.cache else []
        hits = dict(zip(look, self.cache.get_many([captions[i] for i in look]))) if look else {}
        todo = []
        for i, c in enumerate(captions):
            if not c:
                out[i] = torch.zeros(0, self.dim, dtype=torch.float16)
            elif c in self.memo:
                out[i] = self.memo[c]
            elif hits.get(i) is not None:
                out[i] = hits[i]
            else:
                todo.append(i)
        if todo:
            self.misses += len(todo)
            h, mask = self._llm()([captions[i] for i in todo])
            for j, i in enumerate(todo):
                out[i] = h[j, : int(mask[j].sum())].to("cpu", torch.float16)
                if len(self.memo) < 2000:  # typed prompts, preview prompts
                    self.memo[captions[i]] = out[i]
        return out

    def __call__(self, captions):
        captions = list(captions)
        _, _, pooled = self.clip(captions)
        toks = self.tokens(captions)
        T = self.spec["max_tokens"] - 1  # always the full length: a per-batch length makes the compiled UNet's
        # shapes dynamic, and Inductor's backward for that is wrong (torch 2.10: "shape '[164, 1024]' is invalid")
        ctx = torch.zeros(len(toks), T, self.dim, dtype=torch.float16)
        mask = torch.zeros(len(toks), T, dtype=torch.long)
        for i, t in enumerate(toks):
            ctx[i, : len(t)] = t
            mask[i, : len(t)] = 1
        return ctx.to(self.device, non_blocking=True), mask.to(self.device), pooled

    def feature_stats(self, captions):
        """Per-dim mean and std of the token states for TextAdapter.set_stats: from the cache (all its tokens),
        else from `captions`."""
        if self.cache:
            return self.cache.mean, self.cache.std
        captions = [c for c in captions if c]
        x = torch.cat([t for i in range(0, len(captions), 128) for t in self.tokens(captions[i:i + 128])]).float()
        return x.mean(0), x.std(0)

    def param_count(self):
        n = sum(p.numel() for p in self.clip.model.parameters())
        if self.llm is not None:
            return n + sum(p.numel() for p in self.llm.model.parameters())
        return n + self.cache.meta.get("llm_params", 0)


def build_text_encoder(cfg, device):
    """CLIP only (a config without llm_model), or LLM tokens + CLIP pooled when the config has llm_model."""
    if cfg.get("llm_model"):
        return LLMText(cfg, device)
    return TextEncoder(cfg["text_model"], device, cfg.get("max_text_tokens", 77))
