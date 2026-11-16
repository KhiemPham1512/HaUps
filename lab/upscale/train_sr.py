"""Fine-tune the production anime upscaler (RealESRGAN_x4plus_anime_6B, RRDB 6 blocks, x4)
so it keeps faint source details (sweat streaks, blush, soft highlights) that the stock
model cleans away. Fidelity-oriented: mild degradations matched to clean digital art and
pixel + frequency losses (no GAN).

Pairs are synthetic: a 256 px HR patch of an approved image is the target, its degraded
x4-downscaled copy is the input. Images listed in --exclude-manifest (frozen eval set) are
never used.

CLI:
  python lab/upscale/train_sr.py RUN_NAME --char-dir DIR [--char-dir DIR ...] --bg-dir DIR
      --exclude-manifest MANIFEST [--photo-dir DIR] [--photo-share 0.0]
      [--steps 2000] [--batch 8] [--lr 5e-5] [--seed 0] [--probe]
Output: .local-artifacts/upscale-train/runs/RUN_NAME/{model.safetensors, step-*/, records.json}
"""

import argparse
import hashlib
import io
import json
import random
import re
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".local-artifacts" / "upscale-train"
BASE_MODEL = WORK / "models" / "RealESRGAN_x4plus_anime_6B.pth"
BASE_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
SCALE = 4
PATCH = 256  # HR patch; LR is 64
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CHARACTER_SHARE = 0.5
DOWN_FILTERS = (
    Image.Resampling.BICUBIC,
    Image.Resampling.LANCZOS,
    Image.Resampling.BOX,
    Image.Resampling.BILINEAR,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(directory, excluded_sha256=()):
    """Image files directly inside directory (no recursion, so remove/ is never read)."""
    paths = sorted(
        p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not excluded_sha256:
        return paths
    excluded = set(excluded_sha256)
    return [p for p in paths if sha256_file(p) not in excluded]


def manifest_hashes(manifest_path):
    """All sha256 values found anywhere in the frozen-fixture manifest."""
    text = Path(manifest_path).read_text(encoding="utf-8-sig")
    return set(re.findall(r"\b[0-9a-f]{64}\b", text))


def pick_source_kind(roll, photo_share):
    if roll < CHARACTER_SHARE:
        return "character"
    if roll < CHARACTER_SHARE + photo_share:
        return "photo"
    return "background"


def detail_score(patch):
    """Mean edge magnitude; flat patches score near 0."""
    edges = patch.convert("L").filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    return sum(value * count for value, count in enumerate(histogram)) / (patch.width * patch.height)


def sample_patch(image, rng, size=PATCH, candidates=4):
    """Random HR patch, biased towards detailed areas; random rescale and flip."""
    image = image.convert("RGB")
    limit = min(image.size)
    if limit >= size * 2 and rng.random() < 0.5:
        # scale diversity + cleaner targets (averages source JPEG noise)
        factor = rng.uniform(max(0.5, size * 1.2 / limit), 1.0)
        image = image.resize(
            (max(size, round(image.width * factor)), max(size, round(image.height * factor))),
            Image.Resampling.BOX if factor < 0.7 else Image.Resampling.LANCZOS,
        )
    if min(image.size) < size:
        image = ImageOps.pad(image, (max(size, image.width), max(size, image.height)), color=(255, 255, 255))
    best, best_score = None, -1.0
    for _ in range(candidates if rng.random() < 0.7 else 1):
        x = rng.randint(0, image.width - size)
        y = rng.randint(0, image.height - size)
        patch = image.crop((x, y, x + size, y + size))
        score = detail_score(patch)
        if score > best_score:
            best, best_score = patch, score
    return ImageOps.mirror(best) if rng.random() < 0.5 else best


def degrade(patch, rng):
    """Mild degradation of a clean-art HR patch into its x4 LR input."""
    if rng.random() < 0.3:
        patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    low = patch.resize(
        (patch.width // SCALE, patch.height // SCALE), rng.choice(DOWN_FILTERS)
    )
    roll = rng.random()
    if roll < 0.5:
        buffer = io.BytesIO()
        low.save(buffer, format="JPEG", quality=rng.randint(70, 98))
        low = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    elif roll < 0.6:
        buffer = io.BytesIO()
        low.save(buffer, format="WEBP", quality=rng.randint(75, 95))
        low = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return low


def load_model(torch, path=BASE_MODEL):
    from spandrel import ModelLoader

    descriptor = ModelLoader().load_from_file(str(path))
    if descriptor.scale != SCALE:
        raise SystemExit(f"unexpected model scale {descriptor.scale}")
    return descriptor.model


def to_tensor(torch, images):
    import numpy as np

    array = np.stack([np.asarray(image, dtype=np.float32) / 255.0 for image in images])
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def fidelity_loss(torch, prediction, target, fft_weight=0.1):
    """Charbonnier pixel loss + L1 on the 2-D spectrum (keeps faint high-frequency detail)."""
    prediction, target = prediction.float(), target.float()
    pixel = torch.sqrt((prediction - target) ** 2 + 1e-6).mean()
    spectrum = (
        torch.fft.rfft2(prediction, norm="ortho") - torch.fft.rfft2(target, norm="ortho")
    ).abs().mean()
    return pixel + fft_weight * spectrum, pixel.detach(), spectrum.detach()


VGG_LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}  # conv1_2..conv5_4 pre-ReLU, Real-ESRGAN weights


def load_vgg(torch):
    """Frozen VGG19 feature extractor (torchvision ImageNet weights, cached on drive D)."""
    import os

    os.environ.setdefault("TORCH_HOME", str(WORK / "torch-cache"))  # keep the 548 MB download off drive C
    from torchvision.models import VGG19_Weights, vgg19

    features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(VGG_LAYERS) + 1]
    return features.eval().requires_grad_(False)


def perceptual_loss(torch, vgg, prediction, target):
    mean = torch.tensor([0.485, 0.456, 0.406], device=prediction.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=prediction.device).view(1, 3, 1, 1)
    x, y = (prediction.float() - mean) / std, (target.float() - mean) / std
    loss = 0.0
    for index, layer in enumerate(vgg):
        x, y = layer(x), layer(y)
        if index in VGG_LAYERS:
            loss = loss + VGG_LAYERS[index] * (x - y).abs().mean()
    return loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    parser.add_argument("--char-dir", action="append", required=True)
    parser.add_argument("--bg-dir", required=True)
    parser.add_argument("--exclude-manifest", required=True)
    parser.add_argument("--photo-dir")
    parser.add_argument("--photo-share", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--perceptual", type=float, default=0.0, help="weight of the VGG19 perceptual loss (0 = off)")
    parser.add_argument("--probe", action="store_true", help="20 steps, report VRAM and s/step only")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("invalid run name")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 <= args.photo_share <= 0.5:
        raise SystemExit(f"--photo-share must be within [0, 0.5], got {args.photo_share}")
    photo_paths = []
    if args.photo_share > 0:
        if not args.photo_dir:
            raise SystemExit("--photo-dir is required when --photo-share > 0")
        photo_paths = list_images(args.photo_dir)
        if len(photo_paths) < 20:
            raise SystemExit(f"too few photo images: {len(photo_paths)} (need >= 20)")
    run_dir = WORK / "runs" / args.run_name
    run_dir.mkdir(parents=True)  # never overwrite evidence
    if sha256_file(BASE_MODEL) != BASE_SHA256:
        raise SystemExit("base model hash mismatch")
    excluded = manifest_hashes(args.exclude_manifest)
    char_paths = [p for d in args.char_dir for p in list_images(d, excluded)]
    bg_paths = list_images(args.bg_dir)
    if len(char_paths) < 10 or len(bg_paths) < 100:
        raise SystemExit(f"too few images: characters={len(char_paths)} backgrounds={len(bg_paths)}")
    rng = random.Random(args.seed)
    characters = [Image.open(p).convert("RGB") for p in char_paths]  # small set, keep decoded

    import torch
    from safetensors.torch import save_file

    torch.manual_seed(args.seed)
    device = "cuda"
    model = load_model(torch).to(device).train()
    vgg = load_vgg(torch).to(device) if args.perceptual > 0 else None
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")
    steps = 20 if args.probe else args.steps
    torch.cuda.reset_peak_memory_stats()
    history, started = [], perf_counter()
    for step in range(1, steps + 1):
        targets, inputs = [], []
        for _ in range(args.batch):
            kind = pick_source_kind(rng.random(), args.photo_share)
            if kind == "character":
                source = rng.choice(characters)
            elif kind == "photo":
                with Image.open(rng.choice(photo_paths)) as opened:
                    source = opened.convert("RGB")
            else:
                with Image.open(rng.choice(bg_paths)) as opened:
                    source = opened.convert("RGB")
            patch = sample_patch(source, rng)
            targets.append(patch)
            inputs.append(degrade(patch, rng))
        target = to_tensor(torch, targets).to(device)
        low = to_tensor(torch, inputs).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(low)
        loss, pixel, spectrum = fidelity_loss(torch, prediction, target)
        perceptual = perceptual_loss(torch, vgg, prediction, target) if vgg is not None else torch.zeros((), device=device)
        loss = loss + args.perceptual * perceptual
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}")
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[key].mul_(0.999).add_(value.detach(), alpha=0.001)
                else:
                    ema[key].copy_(value)
        if step % 10 == 0 or step == steps:
            history.append({"step": step, "loss": round(loss.item(), 5), "pixel": round(pixel.item(), 5), "fft": round(spectrum.item(), 5), "perceptual": round(perceptual.item(), 5)})
            print(f"step={step}/{steps} loss={loss.item():.5f} pixel={pixel.item():.5f} fft={spectrum.item():.5f} perc={perceptual.item():.4f}", flush=True)
        if not args.probe and (step % args.save_every == 0 or step == steps):
            target_dir = run_dir if step == steps else run_dir / f"step-{step}"
            target_dir.mkdir(exist_ok=True)
            save_file({k: v.contiguous() for k, v in ema.items()}, str(target_dir / "model.safetensors"))
    seconds = perf_counter() - started
    records = {
        "run_name": args.run_name,
        "base_model": BASE_MODEL.name,
        "base_sha256": BASE_SHA256,
        "architecture": "RRDBNet 6 blocks x4 (spandrel), EMA 0.999 weights saved",
        "loss": f"charbonnier + 0.1 * rfft2 L1 + {args.perceptual} * VGG19 perceptual (conv1_2..conv5_4, 0.1/0.1/1/1/1)",
        "degradation": "x4 random filter; 30% gaussian blur 0.2-0.9; 50% JPEG 70-98; 10% WebP 75-95",
        "steps": steps, "batch": args.batch, "lr": args.lr, "seed": args.seed, "patch": PATCH,
        "character_images": len(char_paths), "background_images": len(bg_paths),
        "excluded_hashes": len(excluded), "character_share": CHARACTER_SHARE,
        "photo_share": args.photo_share, "photos": len(photo_paths),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "seconds": round(seconds, 1), "seconds_per_step": round(seconds / steps, 3),
        "probe": args.probe, "history": history,
    }
    (run_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"peak_vram_mb={records['peak_vram_mb']} s_per_step={records['seconds_per_step']} steps={steps}", flush=True)


if __name__ == "__main__":
    main()
