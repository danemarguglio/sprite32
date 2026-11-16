#!/usr/bin/env python3
"""Phone-first sample generator (scripts/sprite_web.html).

    python scripts/sprite_web.py --port 8077                  # weights/sprite32-1.0 and weights/sprite32-1.0-fast
    python scripts/sprite_web.py full=RUN_DIR fast=RUN_DIR    # other run directories, with the label to show
    (scripts/sprite_web.sh start|stop|status|log runs it in tmux)

Sampling is hf_sample.sample_images with hf_sample.py's defaults (dpmpp_2m, 35 steps, guidance 3, quality 5, no
category); a run with the guidance in its weights (distill_guidance.py) gets 12 steps. The model is loaded once and kept warm; all GPU work runs on one worker thread, so requests are
serialised. After --idle_minutes without a request the UNet, CLIP and Qwen are dropped and the VRAM is freed; the
next request loads them again.

Qwen3-4B is loaded once as AutoModelForCausalLM. Its .model is the same Qwen3Model hf_text.LLMEncoder gets from
AutoModel, so the conditioning states are unchanged (checked with --check_encoder), and the tied LM head shares the
embedding matrix, so "Expand" (short idea -> training-style caption) needs no second model and no extra weights.

With more than one run directory every request is drawn by each model from the same seeds and the page shows
them one under the other. "Expand" always uses the first model's Qwen.

Results go to outputs_diffusers/web_samples/<gen_id>/: NN.png (raw 32x32 straight-alpha RGBA) + meta.json. With
two models, n images each: the first model's are 00..n-1 and the second's n..2n-1 (meta.json "models").
Served: the page, /api/{status,generate,job,expand,surprise,history}, and /img/<gen_id>/<NN>.png[?scale=16].
"""

import argparse
import gc
import json
import random
import re
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import hf_common  # noqa: E402
import hf_text  # noqa: E402
from hf_sample import load_unet, sample_images  # noqa: E402

DEFAULT_RUNS = ["sprite32 1.0=weights/sprite32-1.0", "sprite32 1.0 fast=weights/sprite32-1.0-fast"]
WEIGHTS = "ema0.9999"
OUT_DIR = ROOT / "outputs_diffusers/web_samples"
PROMPTS = Path(__file__).with_name("sprite_web_prompts.txt")
HTML = Path(__file__).with_name("sprite_web.html")
STEPS, GUIDANCE, QUALITY = 35, 3.0, 5  # hf_sample.py defaults
FAST_STEPS = 12  # for a run with the guidance in its weights
MAX_PROMPT, MAX_N, MAX_PENDING = 400, 32, 6
GEN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


# ---------------------------------------------------------------- Qwen: one set of weights, two uses

class SharedLLMEncoder(hf_text.LLMEncoder):
    """hf_text.LLMEncoder with the causal-LM wrapper kept: self.model (used by __call__) is lm.model."""

    def __init__(self, name, device, max_tokens=64, layer=-2, prefix="<|im_start|>"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.padding_side = "right"
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(torch.device(device).type, torch.float32)
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval().requires_grad_(False)
        if self.lm.lm_head.weight.data_ptr() != self.lm.model.embed_tokens.weight.data_ptr():
            print("warning: lm_head is not tied to the embeddings (extra VRAM)", flush=True)
        self.model = self.lm.model
        self.device, self.max_tokens, self.layer, self.prefix = device, max_tokens, layer, prefix
        self.dim = self.model.config.hidden_size


class SharedLLMText(hf_text.LLMText):
    """LLMText whose live LLM is SharedLLMEncoder (LLMText.__init__ calls _llm() when there is no text cache)."""

    def _llm(self):
        if self.llm is None:
            self.llm = SharedLLMEncoder(self.spec["model"], self.device, self.spec["max_tokens"],
                                        self.spec["layer"], self.spec["prefix"])
        return self.llm


# ---------------------------------------------------------------- captions for Surprise / Expand

def run_name(run):
    """What the page is told about a run directory: its path in the repository, else only the folder name."""
    return str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else run.name


def load_bundled_prompts():
    return [line.strip() for line in PROMPTS.read_text().splitlines() if line.strip()]


def load_training_captions(data_dir, n=3000, seed=0):
    """(short, long) caption pairs from n random quality>=4 sidecars listed in data_dir/_index.tsv, cached as
    web_samples/captions.json. Keys are inverted in the sidecars (hf_data.py): description = long.
    Without the training set on disk there are none, and Surprise and Expand do without them."""
    cache = OUT_DIR / "captions.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not (data_dir / "_index.tsv").exists():
        return []
    lines = (data_dir / "_index.tsv").read_text().splitlines()[1:]
    good = [ln.split("\t")[0] for ln in lines if ln.split("\t")[-1] in ("4", "5")]
    pairs = []
    for rel in random.Random(seed).sample(good, min(n, len(good))):
        try:
            meta = json.loads((data_dir / f"{rel}.json").read_text())
        except (OSError, ValueError):
            continue
        short, long = meta.get("verbose_description"), meta.get("description")
        if short and long and not meta.get("has_text") and len(long) < 300:
            pairs.append([short.strip(), long.strip()])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pairs))
    return pairs


EXPAND_SYSTEM = ("You write captions for a 32x32 pixel sprite generator. Given a short idea, write one caption in "
                 "the same style as the examples: one or two plain sentences describing the subject, its colours, "
                 "its pose or shape, the art style (pixel art, flat icon, emoji, cartoon clip art...) and the "
                 "background (a transparent background unless the idea asks for another). Keep it under 45 words. "
                 "Reply with the caption only.")


# ---------------------------------------------------------------- model holder (all methods run on the GPU thread)

class Engine:
    def __init__(self, device, idle_seconds, runs):
        self.runs = runs  # [(label, run dir)]; the first one also serves Expand
        self.device = device
        self.idle_seconds = idle_seconds
        self.nets = []  # one dict per run while loaded: label, run, cfg, unet, text, scheduler, steps
        self.state = "cold"  # cold | loading | ready
        self.last_used = time.time()
        self.pool = ThreadPoolExecutor(max_workers=1)  # the only thread that touches the GPU
        self.pending = 0
        self.lock = threading.Lock()
        self.load_seconds = None

    # -- lifecycle
    def _load(self):
        if self.nets:
            return
        self.state = "loading"
        t = time.time()
        try:
            nets = []
            for i, (label, run) in enumerate(self.runs):
                cfg = hf_common.load_run_meta(run)
                text = (SharedLLMText if i == 0 else hf_text.LLMText)(cfg, self.device)
                nets.append({"label": label, "run": run, "cfg": cfg, "unet": load_unet(run, WEIGHTS, self.device),
                             "text": text, "scheduler": hf_common.build_scheduler(cfg, "dpmpp_2m"),
                             "steps": FAST_STEPS if cfg.get("distilled_guidance") else STEPS})
            self.nets = nets
        except Exception:
            del nets
            gc.collect()
            torch.cuda.empty_cache()
            self.state = "cold"
            raise
        self.load_seconds = round(time.time() - t, 1)
        self.state = "ready"
        print(f"model loaded in {self.load_seconds}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    def _unload(self):
        if not self.nets or time.time() - self.last_used < self.idle_seconds:
            return
        self.nets = []
        gc.collect()
        torch.cuda.empty_cache()
        self.state = "cold"
        print(f"idle {self.idle_seconds}s: model unloaded, {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
              f"allocated", flush=True)

    def submit(self, fn, *args):
        with self.lock:
            if self.pending >= MAX_PENDING:
                raise HTTPException(429, "The GPU queue is full. Try again in a moment.")
            self.pending += 1
        self.last_used = time.time()

        def run():
            try:
                self._load()
                return fn(*args)
            finally:
                self.last_used = time.time()
                with self.lock:
                    self.pending -= 1
        return self.pool.submit(run)

    def idle_watch(self):
        while True:
            time.sleep(10)
            if self.nets and self.pending == 0 and time.time() - self.last_used >= self.idle_seconds:
                self.pool.submit(self._unload)

    # -- work
    def generate(self, job, prompt, n, seed, guidance):
        calls = {"n": 0}
        total = sum(net["steps"] * (2 if guidance != 1.0 and not net["cfg"].get("distilled_guidance") else 1)
                    for net in self.nets)

        class Counting(torch.nn.Module):  # counts UNet calls for the progress bar; sampling is hf_sample's
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.distilled_guidance = getattr(inner, "distilled_guidance", None)  # read by the sampler

            def forward(self, *a, **k):
                calls["n"] += 1
                job["progress"] = min(calls["n"] / total, 1.0)
                return self.inner(*a, **k)

        job["state"] = "running"
        t = time.time()
        xs = []
        for net in self.nets:  # the same seed for every model, so image i starts from the same noise in each
            cfg = net["cfg"]
            labels = hf_common.null_labels(cfg, n, QUALITY, 0)
            xs.append(sample_images(Counting(net["unet"]), net["text"], net["scheduler"], [prompt], n, net["steps"],
                                    guidance, None, seed, None, 0.0, cfg.get("channels", 4),
                                    cfg.get("image_size", 32), labels))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gen_seconds = time.time() - t
        gen_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        d = OUT_DIR / gen_id
        d.mkdir(parents=True)
        for k, x in enumerate(xs):
            for i in range(n):  # to_pil_grid on one image is that image, straight alpha
                hf_common.to_pil_grid(x[i:i + 1], nrow=1, background="alpha").save(d / f"{k * n + i:02d}.png")
        meta = {"id": gen_id, "prompt": prompt, "n": n, "seed": seed, "guidance": guidance,
                "steps": self.nets[0]["steps"], "sampler": "dpmpp_2m", "weights": WEIGHTS,
                "run": run_name(self.nets[0]["run"]),
                "quality": QUALITY, "created": datetime.now().isoformat(timespec="seconds"),
                "gpu_seconds": round(gen_seconds, 2)}
        if len(self.nets) > 1:
            meta["models"] = [{"label": net["label"], "run": run_name(net["run"]), "steps": net["steps"]}
                              for net in self.nets]
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta

    @torch.no_grad()
    def expand(self, idea, examples):
        llm = self.nets[0]["text"]._llm()
        msgs = [{"role": "system", "content": EXPAND_SYSTEM}]
        for short, long in examples:
            msgs += [{"role": "user", "content": short}, {"role": "assistant", "content": long}]
        msgs.append({"role": "user", "content": idea})
        text = llm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        tok = llm.tokenizer(text, return_tensors="pt").to(self.device)
        out = llm.lm.generate(**tok, max_new_tokens=90, do_sample=True, temperature=0.7, top_p=0.9,
                              pad_token_id=llm.tokenizer.eos_token_id)
        cap = llm.tokenizer.decode(out[0, tok.input_ids.shape[1]:], skip_special_tokens=True)
        cap = re.sub(r"<think>.*?</think>", "", cap, flags=re.S).strip().strip('"').strip()
        return " ".join(cap.split())[:MAX_PROMPT]


# ---------------------------------------------------------------- web app

def gpu_mem():
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "device_used_gib": round((total - free) / 2**30, 2)}


class GenerateReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT)
    n: int = Field(16, ge=1, le=MAX_N)
    guidance: float = Field(GUIDANCE, ge=1.0, le=8.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class ExpandReq(BaseModel):
    idea: str = Field(min_length=1, max_length=200)


def make_app(engine, bundled, pairs):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    jobs = {}  # job id -> dict, last 200 kept

    @app.get("/", response_class=HTMLResponse)
    def page():
        return HTMLResponse(HTML.read_text(), headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        return {"state": engine.state, "pending": engine.pending, "idle_minutes": engine.idle_seconds / 60,
                "defaults": {"n": 16 // len(engine.runs), "guidance": GUIDANCE}, "load_seconds": engine.load_seconds, "gpu": gpu_mem(),
                "models": [label for label, _ in engine.runs]}

    @app.post("/api/generate")
    def generate(req: GenerateReq):
        prompt = " ".join(req.prompt.split())
        if not prompt:
            raise HTTPException(422, "Type a prompt first.")
        seed = req.seed if req.seed is not None else random.randrange(2**31)
        job_id = secrets.token_hex(6)
        job = {"id": job_id, "state": "queued", "progress": 0.0, "prompt": prompt}
        job["future"] = engine.submit(engine.generate, job, prompt, req.n, seed, req.guidance)
        jobs[job_id] = job
        for k in list(jobs)[:-200]:
            jobs.pop(k)
        return {"job": job_id}

    @app.get("/api/job/{job_id}")
    def job(job_id: str):
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown job.")
        f = j["future"]
        out = {"state": j["state"], "progress": j["progress"], "engine": engine.state}
        if f.done():
            if f.exception() is not None:
                out.update(state="error", error=str(f.exception())[:300])
            else:
                out.update(state="done", result=f.result())
        elif j["state"] == "queued" and engine.state != "ready":
            out["state"] = "warming"
        return out

    @app.post("/api/expand")
    async def expand(req: ExpandReq):
        import asyncio
        idea = " ".join(req.idea.split())
        examples = random.sample(pairs, 6) if len(pairs) >= 6 else pairs
        fut = engine.submit(engine.expand, idea, examples)
        return {"idea": idea, "caption": await asyncio.wrap_future(fut)}

    @app.get("/api/surprise")
    def surprise():
        if pairs and random.random() < 0.4:
            return {"prompt": random.choice(random.choice(pairs)), "source": "training caption"}
        return {"prompt": random.choice(bundled), "source": "example prompt"}

    @app.get("/api/history")
    def history(limit: int = Query(40, ge=1, le=200), before: str | None = None):
        dirs = sorted((p for p in OUT_DIR.iterdir() if p.is_dir() and GEN_ID.match(p.name)), reverse=True)
        if before and GEN_ID.match(before):
            dirs = [p for p in dirs if p.name < before]
        out = []
        for p in dirs[:limit]:
            try:
                out.append(json.loads((p / "meta.json").read_text()))
            except (OSError, ValueError):
                continue
        return {"items": out, "more": len(dirs) > limit}

    @app.get("/img/{gen_id}/{name}")
    def image(gen_id: str, name: str, scale: int = Query(1, ge=1, le=32)):
        m = re.fullmatch(r"(\d{2})\.png", name)
        if not GEN_ID.match(gen_id) or not m:
            raise HTTPException(404)
        path = OUT_DIR / gen_id / name
        if not path.is_file():
            raise HTTPException(404)
        cache = {"Cache-Control": "public, max-age=31536000, immutable"}
        if scale == 1:
            return FileResponse(path, media_type="image/png", headers=cache)
        img = Image.open(path)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        buf = BytesIO()
        img.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png", headers=cache)

    @app.exception_handler(Exception)
    def oops(_, exc):
        print(f"error: {exc!r}", flush=True)
        return JSONResponse({"detail": "Generation failed on the server; see the log."}, status_code=500)

    return app


def check_encoder(device, run):
    """Max abs difference of the conditioning states, stock hf_text.LLMEncoder vs SharedLLMEncoder."""
    cfg = hf_common.load_run_meta(run)
    spec = (cfg["llm_model"], device, cfg.get("llm_max_tokens", 64), cfg.get("llm_layer", -2),
            cfg.get("llm_prefix", "<|im_start|>"))
    prompts = ["a red apple", "pixel art pelican riding a bicycle",
               "A simple black line icon of a magnifying glass, on a transparent background.", ""]
    stock = hf_text.LLMEncoder(*spec)
    a, ma = stock(prompts)
    a, ma = a.float().cpu(), ma.cpu()
    del stock
    gc.collect()
    torch.cuda.empty_cache()
    shared = SharedLLMEncoder(*spec)
    b, mb = shared(prompts)
    print(f"shapes {tuple(a.shape)} vs {tuple(b.shape)}; masks equal: {torch.equal(ma, mb.cpu())}; "
          f"max abs diff {(a - b.float().cpu()).abs().max().item():.3g}; "
          f"lm_head tied: {shared.lm.lm_head.weight.data_ptr() == shared.model.embed_tokens.weight.data_ptr()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS, metavar="[LABEL=]RUN_DIR",
                    help="run directories to draw with; relative ones are looked up from the repo")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--idle_minutes", type=float, default=15.0)
    ap.add_argument("--no_preload", action="store_true")
    ap.add_argument("--no_compare", action="store_true", help="serve the first run directory only")
    ap.add_argument("--check_encoder", action="store_true", help="compare conditioning states and exit")
    args = ap.parse_args()
    device = hf_common.pick_device("auto")
    runs = []
    for spec in args.runs:
        label, _, path = spec.rpartition("=")
        run = Path(path) if Path(path).is_absolute() else ROOT / path
        if (run / "run_config.json").exists():
            runs.append((label or run.name, run))
        elif args.runs is not DEFAULT_RUNS or not runs:
            sys.exit(f"{run} has no run_config.json")
    if args.no_compare:
        runs = runs[:1]
    if args.check_encoder:
        return check_encoder(device, runs[0][1])

    cfg = hf_common.load_run_meta(runs[0][1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundled = load_bundled_prompts()
    pairs = load_training_captions(ROOT / cfg["data_dir"])
    print(f"{len(bundled)} example prompts, {len(pairs)} training caption pairs", flush=True)
    print("serving " + ", ".join(f"{label}: {run}" for label, run in runs), flush=True)
    engine = Engine(device, args.idle_minutes * 60, runs)
    if not args.no_preload:
        engine.submit(lambda: None)
    threading.Thread(target=engine.idle_watch, daemon=True).start()

    import uvicorn
    uvicorn.run(make_app(engine, bundled, pairs), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
