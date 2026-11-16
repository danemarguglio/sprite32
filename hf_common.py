"""Shared pieces of the pipeline (hf_train.py, hf_sample.py, hf_eval.py).

Everything model-, scheduler- and EMA-shaped comes from libraries:
  - diffusers.UNet2DConditionModel   (the denoiser)
  - diffusers EDM schedulers         (EDM preconditioning, sigma schedule, solvers)
  - transformers CLIPTextModel       (frozen text encoder)
The custom pieces are the config loader, a 10-line text-encoder wrapper, the optional EDM2
loss-uncertainty head (UncertaintyHead), and these options, all off unless set in the config:
  - qk_norm        cosine attention: RMS-normalised q and k, no learned gain (bounded logits)
  - conv_in_fp32   first conv outside autocast (bf16 quantises the low-noise input)
  - text_adapter   trainable adapter over frozen LLM token states (hf_text.py: Qwen3) for cross-attention
  - lr_schedule "constant_cosine_tail"
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from diffusers import EDMDPMSolverMultistepScheduler, EDMEulerScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import GaussianFourierProjection
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizerFast
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # CLIPTextModel from the full CLIP checkpoint lists unused vision keys


def load_config(path, overrides=()):
    """YAML config (optional `base:` file it extends) plus `key=value` overrides."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("base"):
        base = load_config(Path(path).parent / cfg.pop("base"))
        cfg = {**base, **cfg}
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


# Label vocabulary of the sidecars (quality is 1-5). Index 0 = "no label" (dropped/unknown).
CATEGORIES = ["icon", "emoji", "game_sprite", "pixel_art", "logo", "text", "ui_element",
              "tile_or_texture", "photo", "scene", "abstract", "blank_or_noise", "other"]
N_QUALITY = 5


def label_dim(cfg):
    """Width of the quality + category one-hots appended to the pooled text vector (0 without label_cond)."""
    return (N_QUALITY + 1) + (len(CATEGORIES) + 1) if cfg.get("label_cond") else 0


def label_index(meta):
    """(quality, category) indices from a sidecar dict; 0 where missing or unknown."""
    q = meta.get("quality")
    q = q if isinstance(q, int) and 1 <= q <= N_QUALITY else 0
    c = meta.get("category")
    return q, CATEGORIES.index(c) + 1 if c in CATEGORIES else 0


def add_labels(pooled, labels):
    """Append quality/category one-hots to pooled text embeddings. labels: None (model has no label input)
    or a (quality, category) pair of LongTensors of shape (B,), 0 = null."""
    if labels is None:
        return pooled
    q, c = (t.to(pooled.device) for t in labels)
    onehot = torch.cat([nn.functional.one_hot(q, N_QUALITY + 1), nn.functional.one_hot(c, len(CATEGORIES) + 1)], 1)
    return torch.cat([pooled, onehot.to(pooled.dtype)], 1)


def null_labels(cfg, n, quality=0, category=0):
    """Labels for sampling: None for a model without label_cond, else constant (quality, category) indices."""
    if not cfg.get("label_cond"):
        return None
    return torch.full((n,), quality, dtype=torch.long), torch.full((n,), category, dtype=torch.long)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QKNorm(nn.Module):
    """RMS-normalise the last (per-head channel) dim, no learned gain: EDM2's cosine attention. With q and k both
    normalised, a logit q.k / sqrt(d) lies in [-scale * sqrt(d), scale * sqrt(d)] whatever the weights do, so the
    failure of an earlier run (logits of one 32x32 self-attention layer growing to 1e6) cannot happen.
    scale 1 is EDM2 (+-8 at its d = 64 with <= 256 keys); here d = 32 with 1024 keys, see configs/sprite32.yaml."""

    def __init__(self, scale=1.0, eps=1e-6):
        super().__init__()
        self.gain, self.eps = scale ** 0.5, eps

    def forward(self, x):
        r = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.gain
        return x * r.to(x.dtype)


def apply_qk_norm(model, scale=1.0):
    """Give every diffusers Attention in `model` QKNorm on q and k (AttnProcessor2_0 applies attn.norm_q / norm_k
    per head). No parameters, so state dicts are unchanged; build_unet re-applies it from the run config.
    Checked here rather than trusted: the output of one layer must not change when its to_q is scaled."""
    attns = [m for m in model.modules() if isinstance(m, Attention)]
    for m in attns:
        m.norm_q, m.norm_k = QKNorm(scale), QKNorm(scale)
    m = min(attns, key=lambda a: a.to_q.in_features)
    was_training = m.training
    m.eval()  # no dropout in the comparison
    with torch.no_grad():
        x = torch.randn(2, 16, m.to_q.in_features)
        ctx = None if m.to_k.in_features == m.to_q.in_features else torch.randn(2, 5, m.to_k.in_features)
        a = m(x, encoder_hidden_states=ctx)
        m.to_q.weight.mul_(7.0)
        b = m(x, encoder_hidden_states=ctx)
        m.to_q.weight.div_(7.0)
    m.train(was_training)
    if m.to_q.bias is not None or not torch.allclose(a, b, atol=1e-4, rtol=1e-3):
        raise RuntimeError("qk_norm is not applied by this diffusers version's attention processor")
    return model


class Conv2dFP32(nn.Conv2d):
    """Conv2d that runs in fp32 inside autocast. For conv_in: at sigma 0.01 the bf16 cast of the network input
    (8 mantissa bits) is as large as the noise; fp32 here took val loss at sigma 0.01 from 0.222 to 0.196."""

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())


class TextAdapter(nn.Module):
    """Frozen LLM token states -> cross-attention context. Per-dim standardisation with fixed statistics (a few
    of Qwen's 2560 dims have 10x the typical std), Linear, a learned first token (always unmasked: it is the whole
    context of an empty caption), learned positions, and `layers` bidirectional transformer blocks: the states of a
    causal LM only see the tokens before them."""

    def __init__(self, in_dim, dim, layers=2, head_dim=64, max_tokens=64):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.proj = nn.Linear(in_dim, dim)
        self.first = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos = nn.Parameter(torch.zeros(max_tokens + 1, dim))
        self.blocks = nn.ModuleList(BasicTransformerBlock(dim, dim // head_dim, head_dim) for _ in range(layers))
        self.norm_out = nn.LayerNorm(dim)

    def set_stats(self, mean, std):
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp(min=1e-3))

    def forward(self, ctx, mask):
        """ctx (B, T, in_dim), mask (B, T) with 1 = token -> (B, T + 1, dim), (B, T + 1)."""
        x = ((ctx.to(self.feat_mean.dtype) - self.feat_mean) / self.feat_std).clamp(-10.0, 10.0)
        x = self.proj(x)
        x = torch.cat([self.first.to(x.dtype).expand(x.shape[0], 1, -1), x], 1) + self.pos[: x.shape[1] + 1].to(x.dtype)
        mask = torch.cat([mask.new_ones(mask.shape[0], 1), mask], 1)
        bias = (1 - mask.to(x.dtype))[:, None, :] * -10000.0
        for block in self.blocks:
            x = block(x, attention_mask=bias)
        return self.norm_out(x), mask


class Sprite32UNet(UNet2DConditionModel):
    """UNet2DConditionModel plus the optional `text_adapter` submodule (attached by build_unet), so the adapter
    is trained, EMA'd, compiled and saved with the UNet. Same constructor and config as the parent."""

    def forward(self, sample, timestep, encoder_hidden_states, *args, encoder_attention_mask=None, **kwargs):
        adapter = self._modules.get("text_adapter")
        if adapter is not None:
            encoder_hidden_states, encoder_attention_mask = adapter(encoder_hidden_states, encoder_attention_mask)
        return super().forward(sample, timestep, encoder_hidden_states, *args,
                               encoder_attention_mask=encoder_attention_mask, **kwargs)


def build_unet(cfg):
    """One level per entry of block_out_channels, attention + cross-attention at every level, pooled
    CLIP text added to the time embedding (class_embed_type="projection"), Fourier
    noise embedding of c_noise = log(sigma)/4 (EDM). Plus the config's options (module docstring)."""
    ch = list(cfg["block_out_channels"])
    n = len(ch)
    unet = Sprite32UNet(
        sample_size=cfg.get("image_size", 32),
        in_channels=cfg.get("channels", 4),
        out_channels=cfg.get("channels", 4),
        down_block_types=("CrossAttnDownBlock2D",) * n,
        up_block_types=("CrossAttnUpBlock2D",) * n,
        block_out_channels=ch,
        layers_per_block=cfg.get("layers_per_block", 2),
        # diffusers naming quirk: in UNet2DConditionModel `attention_head_dim` is the
        # number of heads per level.
        attention_head_dim=[max(c // cfg.get("channels_per_head", 64), 1) for c in ch],
        cross_attention_dim=cfg.get("cross_attention_dim", cfg.get("text_embed_dim", 512)),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        dropout=cfg.get("dropout", 0.0),
        time_embedding_type="fourier",          # GaussianFourierProjection, like EDM2's MPFourier
        time_embedding_dim=cfg.get("time_embedding_dim", ch[-1]),
        class_embed_type="projection",          # pooled CLIP EOS (+ label one-hots) -> embedding stream
        projection_class_embeddings_input_dim=cfg.get("text_embed_dim", 512) + label_dim(cfg),
    )
    if cfg.get("text_adapter"):
        unet.text_adapter = TextAdapter(cfg["text_adapter_in_dim"], cfg["cross_attention_dim"],
                                        cfg.get("text_adapter_layers", 2), max_tokens=cfg.get("llm_max_tokens", 64))
    if cfg.get("qk_norm"):
        apply_qk_norm(unet, cfg.get("qk_norm_scale", 1.0))
    if cfg.get("conv_in_fp32"):
        unet.conv_in.__class__ = Conv2dFP32
    return unet


def load_unet(path, cfg, device):
    """Weights saved by save_pretrained (hf_train.export) into the architecture the run config describes.
    Not from_pretrained: that rebuilds a plain UNet2DConditionModel, without qk_norm / conv_in_fp32 / the adapter."""
    unet = build_unet(cfg)
    unet.load_state_dict(load_file(Path(path) / "diffusion_pytorch_model.safetensors"), strict=True)
    # distill_guidance.py: this guidance scale is in the weights and the samplers skip the unconditional pass
    unet.distilled_guidance = cfg.get("distilled_guidance")
    return unet.to(device).eval()


def build_lr_scheduler(cfg, optimizer):
    """diffusers' get_scheduler, plus "constant_cosine_tail": linear warmup, constant, then a cosine from
    lr_decay_start (default 3/4 of num_steps) down to lr_final_ratio * lr at num_steps."""
    name, warm, total = cfg.get("lr_schedule", "constant_with_warmup"), cfg.get("warmup_steps", 0), cfg["num_steps"]
    if name != "constant_cosine_tail":
        from diffusers.optimization import get_scheduler
        return get_scheduler(name, optimizer, num_warmup_steps=warm, num_training_steps=total)
    start, floor = cfg.get("lr_decay_start", int(0.75 * total)), cfg.get("lr_final_ratio", 0.1)

    def factor(step):
        if step < warm:
            return step / max(1.0, warm)
        if step <= start:
            return 1.0
        t = min((step - start) / max(1, total - start), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def freeze_conv1_bias(unet):
    """Zero and freeze every ResnetBlock2D conv1.bias. It is redundant: time_emb_proj's bias is added
    to the same activations right after conv1, so the net computes the same functions without it.
    Works around an Inductor bug (torch 2.10, bf16): with widths 128/256/512 the compiled backward
    gives non-finite conv1.bias grads in the 256-channel blocks (which block varies run to run);
    eager and aot_eager are clean. Saved checkpoints keep the (zero) bias, so loading is unchanged."""
    from diffusers.models.resnet import ResnetBlock2D
    for m in unet.modules():
        if isinstance(m, ResnetBlock2D) and m.conv1.bias is not None:
            nn.init.zeros_(m.conv1.bias)
            m.conv1.bias.requires_grad_(False)


def build_scheduler(cfg, kind=None):
    """EDM scheduler carrying sigma_data; used for training preconditioning and sampling."""
    kind = kind or cfg.get("sampler", "dpmpp_2m")
    common = dict(
        sigma_min=cfg.get("sigma_min", 0.002),
        sigma_max=cfg.get("sigma_max", 80.0),
        sigma_data=cfg["sigma_data"],
        sigma_schedule="karras",
        rho=cfg.get("rho", 7.0),
        prediction_type="epsilon",  # EDM's F_theta parameterisation (c_out = sigma*sd/sqrt(...))
        final_sigmas_type="zero",
    )
    if kind == "euler":
        return EDMEulerScheduler(**common)
    if kind == "dpmpp_2m":
        return EDMDPMSolverMultistepScheduler(**common, algorithm_type="dpmsolver++", solver_order=2)
    if kind == "dpmpp_2m_sde":
        return EDMDPMSolverMultistepScheduler(**common, algorithm_type="sde-dpmsolver++", solver_order=2)
    raise ValueError(f"unknown sampler {kind}")


class TextEncoder:
    """Frozen CLIP text encoder: per-token states, attention mask, pooled EOS state."""

    def __init__(self, name, device, max_tokens=77):
        self.tokenizer = CLIPTokenizerFast.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name).to(device).eval().requires_grad_(False)
        self.device = device
        self.max_tokens = max_tokens

    @torch.no_grad()
    def __call__(self, captions):
        tok = self.tokenizer(list(captions), padding="max_length", max_length=self.max_tokens,
                             truncation=True, return_tensors="pt").to(self.device)
        out = self.model(input_ids=tok.input_ids, attention_mask=tok.attention_mask)
        return out.last_hidden_state, tok.attention_mask, out.pooler_output


class UncertaintyHead(nn.Module):
    """EDM2 learned loss weighting u(sigma): L = E[loss/exp(u) + u]. Fourier features
    of c_noise -> one linear, zero-init; clamped, since an unbounded u gave NaN losses late in a run."""

    def __init__(self, features=64):
        super().__init__()
        self.fourier = GaussianFourierProjection(features, set_W_to_weight=False, log=False)
        self.linear = nn.Linear(2 * features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c_noise):
        return self.linear(self.fourier(c_noise)).squeeze(-1).clamp(-6.0, 3.0)


def save_run_meta(out_dir, cfg):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "run_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)


def load_run_meta(run_dir):
    with open(Path(run_dir) / "run_config.json") as fh:
        cfg = json.load(fh)
    # a text encoder that ships inside the run directory (the fast model) is named relative to it
    llm = cfg.get("llm_model")
    if llm and (Path(run_dir) / llm / "config.json").exists():
        cfg["llm_model"] = str(Path(run_dir) / llm)
    return cfg


def to_pil_grid(x, nrow, background="white"):
    """x: (B,4,H,W) premultiplied RGBA in [-1,1] -> PIL grid, composited over white
    or kept as straight-alpha RGBA."""
    import torchvision.utils as vutils
    from PIL import Image

    x = ((x.float().cpu() + 1) / 2).clamp(0, 1)
    if x.shape[1] == 4:
        rgb, a = x[:, :3], x[:, 3:4]
        if background == "white":
            x = (rgb + (1 - a)).clamp(0, 1)
        else:  # un-premultiply for a transparent PNG
            x = torch.cat([(rgb / a.clamp(min=1e-3)).clamp(0, 1), a], 1)
    grid = vutils.make_grid(x, nrow=nrow, padding=2, pad_value=1.0 if background == "white" else 0.0)
    arr = (grid.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr, mode="RGBA" if arr.shape[2] == 4 else "RGB")
