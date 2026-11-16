#!/usr/bin/env python3
"""Page for scripts/consistent.py (scripts/consistent_web.html): a template, a word or a list of items, and
matching sprites back.

    python scripts/consistent_web.py RUN_DIR --steps 12 --port 8072

Every request draws the set for --tries seeds (consistent.tried_sets) and the page shows the picked one first;
the other seeds are listed under it and can be chosen by hand. One row with a separate seed per sprite is drawn
for comparison. The model is loaded at the first request and dropped after --idle_minutes; all GPU work runs
on one thread.

Results go to outputs_diffusers/web_consistent/<id>/: set_<seed>.png and lock_<seed>.png (strips of raw 32x32
straight-alpha sprites; lock = identical outside the part that changes), plain.png and meta.json.
Served: the page (/#<id> opens a saved set), /api/{status,make,job,history,set/<id>} and /img/<id>/<name>.png[?scale=8].
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
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import hf_common  # noqa: E402
from consistent import Sampler, item_prompts, lock, strip_image, tried_sets  # noqa: E402

OUT_DIR = ROOT / "outputs_diffusers/web_consistent"
HTML = Path(__file__).with_name("consistent_web.html")
MAX_TEMPLATE, MAX_ITEMS, MAX_ITEM, MAX_TRIES, MAX_PENDING = 300, 12, 40, 16, 4
GEN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
IMG = re.compile(r"^((set|lock)_\d{1,10}|plain)\.png$")


class Engine:
    """Holds the model; every method that touches the GPU runs on the one worker thread."""

    def __init__(self, run_dir, label, device, steps, guidance, weights, idle_seconds):
        self.run_dir, self.device, self.steps, self.guidance, self.weights = run_dir, device, steps, guidance, weights
        self.label = label
        self.idle_seconds = idle_seconds
        self.S = None
        self.state = "cold"  # cold | loading | ready
        self.last_used = time.time()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = 0
        self.lock = threading.Lock()

    def _load(self):
        if self.S is None:
            self.state = "loading"
            try:
                self.S = Sampler(self.run_dir, self.device, self.weights, steps=self.steps, guidance=self.guidance)
            except Exception:
                self.state = "cold"
                raise
            self.state = "ready"

    def _unload(self):
        if self.S is None or time.time() - self.last_used < self.idle_seconds:
            return
        self.S = None
        gc.collect()
        torch.cuda.empty_cache()
        self.state = "cold"
        print(f"idle {self.idle_seconds:.0f}s: model unloaded", flush=True)

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
            if self.S is not None and self.pending == 0 and time.time() - self.last_used >= self.idle_seconds:
                self.pool.submit(self._unload)

    def make(self, job, template, items, letters, tries, share, seed):
        S = self.S
        job["state"] = "running"
        t = time.time()
        prompts, others = item_prompts(template, items, letters)
        plain = S.sample(prompts, S.noise(seed, len(prompts)))  # a separate seed per sprite, for comparison
        sets = tried_sets(S, prompts, others, seed, tries, share,
                          progress=lambda p: job.update(progress=p))
        gen_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        d = OUT_DIR / gen_id
        d.mkdir(parents=True)
        strip_image(plain).save(d / "plain.png")
        for c in sets:
            strip_image(c["set"]).save(d / f"set_{c['seed']}.png")
            strip_image(lock(c["set"], c["mask"])).save(d / f"lock_{c['seed']}.png")
        meta = {"id": gen_id, "template": template, "items": items, "letters": letters, "tries": tries,
                "share": share, "seed": seed, "steps": self.steps, "model": self.label,
                "created": datetime.now().isoformat(timespec="seconds"), "gpu_seconds": round(time.time() - t, 2),
                "sets": [{k: c[k] for k in ("seed", "fit", "px", "margin", "picked")} for c in sets]}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta


class MakeReq(BaseModel):
    template: str = Field(min_length=3, max_length=MAX_TEMPLATE)
    word: str | None = Field(None, max_length=MAX_ITEMS * 2)
    items: list[str] | None = Field(None, max_length=MAX_ITEMS)
    tries: int = Field(8, ge=1, le=MAX_TRIES)
    share: float = Field(5.0, ge=0.5, le=40.0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1 - MAX_TRIES)


def make_app(engine):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    jobs = {}  # job id -> dict, last 100 kept

    @app.get("/", response_class=HTMLResponse)
    def page():
        return HTMLResponse(HTML.read_text(), headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status():
        return {"state": engine.state, "pending": engine.pending, "model": engine.label, "steps": engine.steps,
                "max_items": MAX_ITEMS, "max_tries": MAX_TRIES}

    @app.post("/api/make")
    def make(req: MakeReq):
        template = " ".join(req.template.split())
        if template.count("{}") != 1:
            raise HTTPException(422, "The template needs exactly one {} where the letter or item goes.")
        if req.items:
            items = [" ".join(i.split())[:MAX_ITEM] for i in req.items]
            items, letters = [i for i in items if i], False
        else:
            items, letters = [c for c in (req.word or "") if not c.isspace()], True
        if not 2 <= len(items) <= MAX_ITEMS:
            raise HTTPException(422, f"Give 2 to {MAX_ITEMS} letters or items.")
        seed = req.seed if req.seed is not None else random.randrange(2**31 - 1 - MAX_TRIES)
        job_id = secrets.token_hex(6)
        job = {"id": job_id, "state": "queued", "progress": 0.0}
        job["future"] = engine.submit(engine.make, job, template, items, letters, req.tries, req.share, seed)
        jobs[job_id] = job
        for k in list(jobs)[:-100]:
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
                print(f"error: {f.exception()!r}", flush=True)
                out.update(state="error", error="Generation failed on the server; see the log.")
            else:
                out.update(state="done", result=f.result())
        elif j["state"] == "queued" and engine.state != "ready":
            out["state"] = "warming"
        return out

    @app.get("/api/history")
    def history(limit: int = Query(20, ge=1, le=100)):
        dirs = sorted((p for p in OUT_DIR.iterdir() if p.is_dir() and GEN_ID.match(p.name)), reverse=True)
        out = []
        for p in dirs[:limit]:
            try:
                out.append(json.loads((p / "meta.json").read_text()))
            except (OSError, ValueError):
                continue
        return {"items": out}

    @app.get("/api/set/{gen_id}")
    def one(gen_id: str):
        path = OUT_DIR / gen_id / "meta.json"
        if not GEN_ID.match(gen_id) or not path.is_file():
            raise HTTPException(404, "Unknown set.")
        return json.loads(path.read_text())

    @app.get("/img/{gen_id}/{name}")
    def image(gen_id: str, name: str, scale: int = Query(1, ge=1, le=16)):
        if not GEN_ID.match(gen_id) or not IMG.match(name):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps", type=int, default=35, help="12 is enough for the fast model")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--weights", default="ema0.9999")
    ap.add_argument("--label", default=None, help="model name shown on the page (default: the folder name)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8072)
    ap.add_argument("--idle_minutes", type=float, default=15.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).resolve()
    engine = Engine(run_dir, args.label or run_dir.name, hf_common.pick_device("auto"), args.steps,
                    args.guidance, args.weights, args.idle_minutes * 60)
    print(f"serving {args.run_dir} at {args.steps} steps on port {args.port}", flush=True)
    threading.Thread(target=engine.idle_watch, daemon=True).start()

    import uvicorn
    uvicorn.run(make_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
