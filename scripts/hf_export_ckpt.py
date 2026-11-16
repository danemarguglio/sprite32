#!/usr/bin/env python3
"""Export an accelerate checkpoint (checkpoints/step_N) of an hf_train.py run as a run dir
(unet_raw/, unet_ema<decay>/, scheduler/, run_config.json) that hf_eval.py / hf_sample.py read.

  python scripts/hf_export_ckpt.py outputs_diffusers/sprite32 step_0100000 outputs_diffusers/sprite32_100k
"""
import json, shutil, sys
from pathlib import Path

import torch
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_common import build_scheduler, build_unet  # noqa: E402

run, step, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = json.load(open(run / "run_config.json"))
ck = run / "checkpoints" / step
out.mkdir(parents=True, exist_ok=True)
unet = build_unet(cfg)
sd = {k.removeprefix("_orig_mod."): v for k, v in load_file(ck / "model.safetensors").items()}  # torch.compile prefix
unet.load_state_dict(sd, strict=True)
unet.save_pretrained(out / "unet_raw")
for i, d in enumerate(cfg["ema_decays"]):  # registered for checkpointing in this order
    ema = EMAModel(unet.parameters(), decay=d, foreach=True)
    ema.load_state_dict(torch.load(ck / f"custom_checkpoint_{i}.pkl", weights_only=False))
    assert abs(ema.decay - d) < 1e-12, (ema.decay, d)
    ema.store(unet.parameters()); ema.copy_to(unet.parameters())
    unet.save_pretrained(out / f"unet_ema{d}")
    ema.restore(unet.parameters())
    print(f"ema {d}: optimization_step {ema.optimization_step}")
build_scheduler(cfg).save_pretrained(out / "scheduler")
shutil.copy(run / "run_config.json", out / "run_config.json")
print("exported", ck, "->", out)
