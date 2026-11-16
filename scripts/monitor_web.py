#!/usr/bin/env python3
"""Sprite32 Web Training Monitor — lightweight HTTP server for monitoring training runs."""

import argparse
import json
import math
import os
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs


def _finite(x):
    """NaN/inf -> None: browsers' JSON.parse rejects Python's NaN literal."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    return x


def log_file(run_dir: Path) -> Path:
    """hf_train.py writes training_log.jsonl; older runs wrote training_log.json (both one JSON object per line)."""
    old = run_dir / "training_log.json"
    return old if old.exists() else run_dir / "training_log.jsonl"


def samples_dir(run_dir: Path) -> Path:
    """hf_train.py: previews/ (its samples/ holds hf_sample.py grids); older runs: samples/."""
    prev = run_dir / "previews"
    return prev if prev.is_dir() and any(prev.iterdir()) else run_dir / "samples"


def normalize(entry: dict) -> dict:
    """hf_train.py val records {"step", "val": {..., "val_mean", "train_mean"}} -> val_loss for the frontend."""
    val = entry.get("val")
    if isinstance(val, dict) and "val_mean" in val:
        entry = {"step": entry.get("step"), "val_loss": val["val_mean"], "val_train_loss": val.get("train_mean")}
    return entry


def find_runs(output_dir: Path) -> list[dict]:
    """Scan output_dir for directories that look like training runs."""
    runs = []
    if not output_dir.is_dir():
        return runs
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_log = log_file(entry).exists()
        has_samples = samples_dir(entry).is_dir() and any(samples_dir(entry).iterdir())
        if has_log or has_samples:
            log_or_dir = log_file(entry) if has_log else samples_dir(entry)
            runs.append({
                "name": entry.name,
                "has_log": has_log,
                "has_samples": has_samples,
                "mtime": log_or_dir.stat().st_mtime,
            })
    return runs


def read_log(log_path: Path, since_step: int = -1) -> list[dict]:
    """Read training log, optionally returning only entries after since_step."""
    entries = []
    if not log_path.exists():
        return entries
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = normalize(entry)
            if (entry.get("step") or 0) > since_step:
                entries.append(entry)
    return entries


def list_samples(samples_dir: Path) -> list[str]:
    """List sample PNG filenames sorted newest-first."""
    if not samples_dir.is_dir():
        return []
    files = [f.name for f in samples_dir.iterdir() if f.suffix.lower() == ".png"]
    files.sort(reverse=True)
    return files


def run_status(run_dir: Path) -> dict:
    """Quick status for a run: latest step/loss/speed, active flag, val_loss, fid."""
    log_path = log_file(run_dir)
    status = {
        "active": False, "step": None, "loss": None, "lr": None, "speed": None,
        "val_loss": None, "fid": None, "total_steps": None,
    }

    if not log_path.exists():
        return status

    # Check if log was modified recently (< 120s)
    mtime = log_path.stat().st_mtime
    status["active"] = (time.time() - mtime) < 120

    # Read last chunk to find latest entries of each type
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        if pos == 0:
            return status
        chunk_size = min(pos, 16384)
        f.seek(pos - chunk_size)
        chunk = f.read(chunk_size).decode("utf-8", errors="replace")

    for line in reversed(chunk.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            entry = normalize(json.loads(line))
        except json.JSONDecodeError:
            continue
        # Fill in latest values we haven't found yet
        if status["step"] is None and "loss" in entry:
            status["step"] = entry.get("step")
            status["loss"] = entry.get("loss")
            status["lr"] = entry.get("lr")
            status["speed"] = entry.get("steps_per_sec")
        if status["val_loss"] is None and "val_loss" in entry:
            status["val_loss"] = entry["val_loss"]
        if status["fid"] is None and "fid" in entry:
            status["fid"] = entry["fid"]
        if status["step"] is not None and status["val_loss"] is not None and status["fid"] is not None:
            break

    # total_steps: hf_train.py's run_config.json, else a config whose output_dir is this run
    try:
        status["total_steps"] = json.load(open(run_dir / "run_config.json")).get("num_steps")
        return status
    except (OSError, ValueError):
        pass
    run_name = run_dir.name
    configs_dir = run_dir.parent.parent / "configs"
    if configs_dir.is_dir():
        try:
            import yaml
            for cfg_file in configs_dir.glob("*.yaml"):
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f) or {}
                output_dir = cfg.get("output_dir", "")
                if Path(output_dir).name == run_name:
                    status["total_steps"] = cfg.get("num_steps")
                    break
        except Exception:
            pass

    return status


class MonitorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the training monitor."""

    output_dir: Path = Path("outputs")
    frontend_path: Path = Path("scripts/monitor_frontend.html")

    def log_message(self, format, *args):
        # Suppress access logs — too noisy with polling
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(_finite(data)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path):
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, path: Path):
        if not path.exists():
            self.send_error(404, "Image not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _safe_run_path(self, run_name: str) -> Path | None:
        """Resolve run name to path with traversal protection."""
        # Block path traversal
        if "/" in run_name or "\\" in run_name or ".." in run_name:
            return None
        run_path = self.output_dir / run_name
        # Ensure it's actually inside output_dir
        try:
            run_path.resolve().relative_to(self.output_dir.resolve())
        except ValueError:
            return None
        if not run_path.is_dir():
            return None
        return run_path

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        # Serve frontend
        if path == "" or path == "/":
            self._send_html(self.frontend_path)
            return

        # API: list runs
        if path == "/api/runs":
            self._send_json(find_runs(self.output_dir))
            return

        # API: training log
        if path.startswith("/api/log/"):
            run_name = path[len("/api/log/"):]
            run_path = self._safe_run_path(run_name)
            if not run_path:
                self._send_json({"error": "run not found"}, 404)
                return
            since_step = int(params.get("since_step", ["-1"])[0])
            entries = read_log(log_file(run_path), since_step)
            self._send_json(entries)
            return

        # API: list samples
        if path.startswith("/api/samples/"):
            run_name = path[len("/api/samples/"):]
            run_path = self._safe_run_path(run_name)
            if not run_path:
                self._send_json({"error": "run not found"}, 404)
                return
            self._send_json(list_samples(samples_dir(run_path)))
            return

        # API: run status
        if path.startswith("/api/status/"):
            run_name = path[len("/api/status/"):]
            run_path = self._safe_run_path(run_name)
            if not run_path:
                self._send_json({"error": "run not found"}, 404)
                return
            self._send_json(run_status(run_path))
            return

        # Serve sample images
        if path.startswith("/samples/"):
            parts = path[len("/samples/"):].split("/", 1)
            if len(parts) != 2:
                self.send_error(404)
                return
            run_name, filename = parts
            # Validate filename
            if ".." in filename or "/" in filename or "\\" in filename:
                self.send_error(403, "Forbidden")
                return
            run_path = self._safe_run_path(run_name)
            if not run_path:
                self.send_error(404)
                return
            img_path = samples_dir(run_path) / filename
            # Double-check resolved path is inside samples dir
            try:
                img_path.resolve().relative_to(samples_dir(run_path).resolve())
            except ValueError:
                self.send_error(403, "Forbidden")
                return
            self._send_png(img_path)
            return

        self.send_error(404)

    def do_HEAD(self):
        """Handle HEAD requests (browsers/curl use these)."""
        self.do_GET()


def main():
    parser = argparse.ArgumentParser(description="Sprite32 Web Training Monitor")
    parser.add_argument("--output-dir", default="outputs", help="Training outputs directory")
    parser.add_argument("--port", type=int, default=8070, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    frontend_path = Path(__file__).parent / "monitor_frontend.html"

    if not output_dir.is_dir():
        print(f"Warning: output directory '{output_dir}' does not exist")

    MonitorHandler.output_dir = output_dir
    MonitorHandler.frontend_path = frontend_path.resolve()

    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)  # Safari opens parallel keep-alive sockets
    print(f"Sprite32 Monitor — http://{args.host}:{args.port}")
    print(f"Serving runs from: {output_dir}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
