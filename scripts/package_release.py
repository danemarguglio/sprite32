"""Builds the folder that goes on Hugging Face: one sub-folder per version with only what sampling needs
(the EMA 0.9999 UNet, the scheduler, run_config.json and, for the fast version, its text encoder), next to the
model card and the data notices from release/.

    python scripts/package_release.py --full outputs_diffusers/sprite32/exports/step_0500000 \
        --fast outputs_diffusers/distill/guide3/export_small --out ../sprite32-release
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = "unet_ema0.9999"
DOCS = ["NOTICE.md", "SOURCES.md", "DATASHEET.md", "attribution.tsv.gz", "sources.json", "samples.md",
        "samples.png", "samples_fast.png", "matching_sets.png"]


def package(run_dir, out_dir):
    run_dir = Path(run_dir)
    out_dir.mkdir(parents=True)
    shutil.copytree(run_dir / WEIGHTS, out_dir / WEIGHTS)
    shutil.copytree(run_dir / "scheduler", out_dir / "scheduler")
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg["text_cache_dir"] = None  # the cache of training captions is not shipped
    cfg["ema_decays"] = [float(WEIGHTS.removeprefix("unet_ema"))]  # the one shipped, so it is the default
    llm = Path(cfg["llm_model"])
    if (llm / "config.json").exists():  # a local text encoder: ship it; hf_common.load_run_meta finds it
        shutil.copytree(llm, out_dir / "text_encoder")
        cfg["llm_model"] = "text_encoder"
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="export of sprite32 1.0")
    ap.add_argument("--fast", required=True, help="export of sprite32 1.0 fast")
    ap.add_argument("--out", required=True, help="folder to create")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True)
    package(args.full, out / "sprite32-1.0")
    package(args.fast, out / "sprite32-1.0-fast")
    shutil.copy(REPO / "release" / "MODEL_CARD.md", out / "README.md")
    shutil.copy(REPO / "LICENSE", out / "LICENSE")
    for name in DOCS:
        shutil.copy(REPO / "release" / name, out / name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
