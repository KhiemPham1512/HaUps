"""Build an anime SDXL inpainting UNet by "add difference" (CPU only, ~10 GB RAM):

    merged = animagine + (sdxl_inpaint - sdxl_base)

The inpaint UNet has 9 input channels (4 noise + 1 mask + 4 masked-image latents); for
conv_in the anime delta is added to the 4 noise channels only. Output is a diffusers UNet
folder usable with StableDiffusionXLInpaintPipeline.

CLI: python lab/outpainting/sdxl_merge.py   (writes .local-artifacts/outpainting/sdxl/animagine-inpaint-ad/)
"""

import hashlib
import json
import shutil
from pathlib import Path

SDXL = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting" / "sdxl"
ANIME = SDXL / "animagine-xl-4.0" / "unet" / "diffusion_pytorch_model.safetensors"
INPAINT = SDXL / "sdxl-inpaint-0.1" / "unet" / "diffusion_pytorch_model.fp16.safetensors"
BASE = SDXL / "sdxl-base-1.0" / "unet" / "diffusion_pytorch_model.fp16.safetensors"
OUT = SDXL / "animagine-inpaint-ad"


def merge_tensor(anime, inpaint, base):
    """anime + (inpaint - base) in float32; extra inpaint input channels are kept as is."""
    anime, inpaint, base = anime.float(), inpaint.float(), base.float()
    if anime.shape != base.shape:
        raise ValueError(f"anime/base shape mismatch: {tuple(anime.shape)} vs {tuple(base.shape)}")
    if inpaint.shape == anime.shape:
        return anime + inpaint - base
    if inpaint.dim() != 4 or inpaint.shape[0] != anime.shape[0] or inpaint.shape[2:] != anime.shape[2:]:
        raise ValueError(f"unsupported shape change: {tuple(inpaint.shape)} vs {tuple(anime.shape)}")
    merged = inpaint.clone()
    merged[:, : anime.shape[1]] += anime - base
    return merged


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = OUT / "unet"
    target.mkdir(parents=True)  # never overwrite an existing merge
    merged, widened = {}, []
    with safe_open(ANIME, "pt") as anime, safe_open(INPAINT, "pt") as inpaint, safe_open(BASE, "pt") as base:
        keys = set(inpaint.keys())
        if keys != set(anime.keys()) or keys != set(base.keys()):
            raise SystemExit("UNet key sets differ; refusing to merge")
        for key in sorted(keys):
            a, i = anime.get_tensor(key), inpaint.get_tensor(key)
            if a.shape != i.shape:
                widened.append(key)
            out = merge_tensor(a, i, base.get_tensor(key))
            if not torch.isfinite(out).all():
                raise SystemExit(f"non-finite values in {key}")
            merged[key] = out.half()
    if widened != ["conv_in.weight"]:
        raise SystemExit(f"unexpected widened tensors: {widened}")
    save_file(merged, str(target / "diffusion_pytorch_model.safetensors"))
    shutil.copyfile(INPAINT.parent / "config.json", target / "config.json")
    info = {
        "formula": "animagine + (sdxl_inpaint - sdxl_base); conv_in: delta on first 4 channels",
        "tensors": len(merged),
        "sources_sha256": {p.parent.parent.name: sha256(p) for p in (ANIME, INPAINT, BASE)},
        "merged_sha256": sha256(target / "diffusion_pytorch_model.safetensors"),
    }
    (OUT / "merge.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
