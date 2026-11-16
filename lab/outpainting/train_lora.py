"""Local PEFT LoRA trainer for PowerPaint v2's BrushNet branch.

CLI:
  python lab/outpainting/train_lora.py RUN_NAME --images-dir DIR --exclude-manifest MANIFEST [--steps 600] [--rank 8] [--lr 1e-4] [--grad-accum 4] [--seed 0] [--seg-model ONNX] [--bg-dir DIR] [--plain-prob 0.25] [--probe]
"""

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from PIL import Image, ImageFilter, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[2]
LAB_CODE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LAB_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_CODE_DIR))

from app.lock import single_instance  # noqa: E402
from character_check import compute_character_mask, SEG_SIZE  # noqa: E402
from conditioning_gpu import RUN_NAME_RE, sha256_file  # noqa: E402
import powerpaint_gpu as pp  # noqa: E402

LAB = pp.LAB
LORA_RUNS_DIR_NAME = "lora-runs"
ALLOWED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MIN_TRAINING_IMAGES = 8
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
CHECKPOINT_INTERVAL = 200
LOSS_LOG_INTERVAL = 10
PROBE_STEPS = 3


def validate_run_name(run_name):
    """Validate RUN_NAME against RUN_NAME_RE."""
    if not isinstance(run_name, str) or not RUN_NAME_RE.fullmatch(run_name):
        raise ValueError("Invalid LoRA run name")
    return run_name


def create_lora_run_dir(lab_root, run_name):
    """Create a fresh run directory under LAB/'lora-runs'/RUN_NAME without overwriting."""
    validate_run_name(run_name)
    lab_path = Path(lab_root)
    if lab_path.is_symlink():
        raise ValueError("Symlink not allowed for lab directory")
    lora_runs_dir = lab_path / LORA_RUNS_DIR_NAME
    if lora_runs_dir.is_symlink():
        raise ValueError("Symlink not allowed for lora-runs directory")
    lora_runs_dir.mkdir(parents=True, exist_ok=True)
    resolved_lora_runs = lora_runs_dir.resolve()

    run_dir = lora_runs_dir / run_name
    if run_dir.is_symlink() or run_dir.exists():
        raise FileExistsError("LoRA run directory already exists")
    resolved_run = run_dir.resolve()
    if (
        not resolved_run.is_relative_to(resolved_lora_runs)
        or resolved_run.parent != resolved_lora_runs
    ):
        raise ValueError("LoRA run directory escapes lora-runs")
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _has_remove_directory(path):
    """Return True if any directory component of path is named 'remove' (case-insensitive)."""
    p = Path(path)
    parts = p.parts[:-1] if p.is_file() else p.parts
    return any(part.lower() == "remove" for part in parts)


def load_exclude_manifest(manifest_path):
    """Load frozen evaluation manifest and return (dataset_id, set of excluded sha256 digests)."""
    path = Path(manifest_path)
    if path.is_symlink():
        raise ValueError("Symlink not allowed for exclude manifest")
    if not path.is_file():
        raise FileNotFoundError("Exclude manifest file not found")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("Exclude manifest must be a JSON object")
    dataset_id = manifest.get("dataset")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("Exclude manifest missing dataset id")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Exclude manifest must contain a files list")

    excluded = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Exclude manifest file entry must be an object")
        orig_sha = entry.get("original_file_sha256")
        if isinstance(orig_sha, str) and orig_sha.strip():
            excluded.add(orig_sha.strip().lower())
        file_sha = entry.get("sha256")
        if isinstance(file_sha, str) and file_sha.strip():
            excluded.add(file_sha.strip().lower())
    return dataset_id, excluded


def collect_training_images(
    images_dir, exclude_manifest, min_files=MIN_TRAINING_IMAGES
):
    """Collect valid regular image files directly in images_dir excluding evaluation fixtures."""
    dataset_id, excluded_hashes = load_exclude_manifest(exclude_manifest)
    img_dir = Path(images_dir)
    if img_dir.is_symlink():
        raise ValueError("Symlink not allowed for images directory")
    if not img_dir.is_dir():
        raise FileNotFoundError("Training images directory not found")

    collected = []
    if not any(part.lower() == "remove" for part in img_dir.parts):
        for entry in sorted(img_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_symlink():
                continue
            if any(part.lower() == "remove" for part in entry.parts):
                continue
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            digest = sha256_file(entry).lower()
            if digest in excluded_hashes:
                continue
            collected.append((entry, digest))

    if len(collected) < min_files:
        raise SystemExit(
            f"Require at least {min_files} training images after exclusion, got {len(collected)}"
        )
    return collected, dataset_id


assert SEG_SIZE == 1024
CHAR_DILATE_PX = 8  # keep character edge/halo pixels out of the loss


def latent_loss_weight(weight):
    """768x432 L weight -> float32 (54, 96) array in [0, 1] on the VAE latent grid."""
    import numpy as np

    small = weight.resize(
        (pp.INFERENCE_WIDTH // 8, pp.INFERENCE_HEIGHT // 8), Image.Resampling.BOX
    )
    return np.asarray(small, dtype=np.float32) / 255.0


def _random_16_9_box(size, rng):
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image dimensions")

    if w * 9 < h * 16:
        win_w = w
        win_h = max(1, min(h, round(w * 9 / 16)))
        x0 = 0
        y0 = rng.randint(0, max(0, h - win_h))
    else:
        win_h = h
        win_w = max(1, min(w, round(h * 16 / 9)))
        x0 = rng.randint(0, max(0, w - win_w))
        y0 = 0
    return (x0, y0, x0 + win_w, y0 + win_h)


def crop_random_16_9_window(image, rng, box=None):
    """Extract a random 16:9 window from an RGB PIL image and resize to 768x432 LANCZOS."""
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("Expected an RGB PIL Image")
    box = box or _random_16_9_box(image.size, rng)
    window = image.crop(box)
    return window.resize(
        (pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT), Image.Resampling.LANCZOS
    )


def sample_known_box(rng, width=pp.INFERENCE_WIDTH, height=pp.INFERENCE_HEIGHT):
    """Sample a known-region bounding box: 75% centered vertical strip, 25% one-sided."""
    if rng.random() < 0.75:
        min_w = int(math.ceil(0.25 * width))
        max_w = int(0.60 * width)
        strip_w = rng.randint(min_w, max_w)
        x0 = (width - strip_w) // 2
        return (x0, 0, x0 + strip_w, height), "center_strip"

    side = rng.choice(("left", "right", "top", "bottom"))
    if side in ("left", "right"):
        min_w = int(math.ceil(0.40 * width))
        max_w = int(0.70 * width)
        keep_w = rng.randint(min_w, max_w)
        if side == "left":
            return (0, 0, keep_w, height), "left"
        return (width - keep_w, 0, width, height), "right"

    min_h = int(math.ceil(0.40 * height))
    max_h = int(0.70 * height)
    keep_h = rng.randint(min_h, max_h)
    if side == "top":
        return (0, 0, width, keep_h), "top"
    return (0, height - keep_h, width, height), "bottom"


def generate_training_example(image, rng, char_mask=None):
    """Generate (target, masked_input, mask, metadata) from an oriented RGB image.

    With char_mask (L, same size, 255 = character) also returns loss_weight:
    255 on background, 0 on dilated character pixels, aligned with target.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Expected a PIL Image")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    box = _random_16_9_box(rgb.size, rng)
    target = crop_random_16_9_window(rgb, rng, box)
    loss_weight = None
    if char_mask is not None:
        char = char_mask.crop(box).resize(
            (pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT), Image.Resampling.BILINEAR
        )
        char = char.point(lambda v: 255 if v >= 128 else 0)
        char = char.filter(ImageFilter.MaxFilter(2 * CHAR_DILATE_PX + 1))
        loss_weight = ImageOps.invert(char)
    known_box, mode = sample_known_box(
        rng, pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT
    )
    raw_mask = Image.new("L", (pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT), 255)
    raw_mask.paste(0, known_box)

    flipped = rng.random() < 0.5
    if flipped:
        target = ImageOps.mirror(target)
        raw_mask = ImageOps.mirror(raw_mask)
        if loss_weight is not None:
            loss_weight = ImageOps.mirror(loss_weight)
        x0, y0, x1, y1 = known_box
        known_box = (pp.INFERENCE_WIDTH - x1, y0, pp.INFERENCE_WIDTH - x0, y1)
        if mode == "left":
            mode = "right"
        elif mode == "right":
            mode = "left"

    expanded_mask = pp.expand_mask(raw_mask, pp.OVERLAP_PX)
    masked_input = pp.prepare_brushnet_input(target, expanded_mask)
    return {
        "target": target,
        "masked_input": masked_input,
        "mask": expanded_mask,
        "raw_mask": raw_mask,
        "known_box": known_box,
        "mode": mode,
        "flipped": flipped,
        "loss_weight": loss_weight,
    }


PLAIN_BACKGROUND_PROB = 0.25  # character art on a flat backdrop (the eval-5 failure case)


def sample_frame_box(size, rng, width=pp.INFERENCE_WIDTH, height=pp.INFERENCE_HEIGHT):
    """Mimic the app: a (lightly cropped) source frame contained in the canvas.

    Returns (crop_box in the source frame, known_box on the canvas), same aspect.
    """
    cw, ch = size
    bh = height if rng.random() < 0.7 else rng.randint(340, height)
    lo, hi = 0.25 * width / bh, 0.70 * width / bh
    crop_w, crop_h = cw, ch
    if cw / ch > hi:
        crop_w = max(1, round(ch * hi))
    elif cw / ch < lo:
        crop_h = max(1, round(cw / lo))
    cx0 = rng.randint(0, cw - crop_w)
    cy0 = rng.randint(0, ch - crop_h)
    bw = max(1, min(width, round(bh * crop_w / crop_h)))
    x0 = max(0, min(width - bw, (width - bw) // 2 + rng.randint(-80, 80)))
    y0 = rng.choice((0, height - bh))
    return (cx0, cy0, cx0 + crop_w, cy0 + crop_h), (x0, y0, x0 + bw, y0 + bh)


def synthetic_plain_background(rng):
    """Smooth two-colour gradient backdrop, mostly light."""
    size = (pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT)
    low = 200 if rng.random() < 0.7 else 0
    first = tuple(rng.randint(low, 255) for _ in range(3))
    second = tuple(max(0, min(255, v + rng.randint(-40, 40))) for v in first)
    gradient = (
        Image.linear_gradient("L")
        .rotate(rng.choice((0, 90, 180, 270)))
        .resize(size, Image.Resampling.BILINEAR)
    )
    return Image.composite(
        Image.new("RGB", size, first), Image.new("RGB", size, second), gradient
    )


def generate_composite_example(char_image, char_mask, background, rng):
    """App-like sample whose to-generate region is pure background.

    The character frame fills the known box (so any cut of the character lies on
    the box edge, as in the app), its own background is replaced through
    char_mask by `background` (None = synthetic plain backdrop), and everything
    outside the known box is that same background.
    """
    size = (pp.INFERENCE_WIDTH, pp.INFERENCE_HEIGHT)
    if background is None:
        target = synthetic_plain_background(rng)
    else:
        target = crop_random_16_9_window(background.convert("RGB"), rng)
    crop_box, known_box = sample_frame_box(char_image.size, rng)
    box_size = (known_box[2] - known_box[0], known_box[3] - known_box[1])
    frame = char_image.convert("RGB").crop(crop_box).resize(
        box_size, Image.Resampling.LANCZOS
    )
    # erode 1 px so the cut-out carries no fringe of its original background
    alpha = (
        char_mask.crop(crop_box)
        .resize(box_size, Image.Resampling.BILINEAR)
        .filter(ImageFilter.MinFilter(3))
    )
    target.paste(frame, known_box[:2], alpha)
    raw_mask = Image.new("L", size, 255)
    raw_mask.paste(0, known_box)

    flipped = rng.random() < 0.5
    if flipped:
        target = ImageOps.mirror(target)
        raw_mask = ImageOps.mirror(raw_mask)
        x0, y0, x1, y1 = known_box
        known_box = (size[0] - x1, y0, size[0] - x0, y1)

    expanded_mask = pp.expand_mask(raw_mask, pp.OVERLAP_PX)
    return {
        "target": target,
        "masked_input": pp.prepare_brushnet_input(target, expanded_mask),
        "mask": expanded_mask,
        "raw_mask": raw_mask,
        "known_box": known_box,
        "mode": "composite",
        "flipped": flipped,
        "loss_weight": None,
    }


def _resolve_device_and_dtype(pipe, torch_mod):
    device = None
    dtype = None
    for comp in (
        getattr(pipe, "brushnet", None),
        getattr(pipe, "vae", None),
        getattr(pipe, "unet", None),
    ):
        if comp is None:
            continue
        if device is None:
            dev = getattr(comp, "device", None)
            if dev is None and hasattr(comp, "parameters"):
                params = list(comp.parameters())
                if params:
                    dev = params[0].device
            if dev is not None:
                device = dev
        if dtype is None:
            dt = getattr(comp, "dtype", None)
            if dt is None and hasattr(comp, "parameters"):
                params = list(comp.parameters())
                if params:
                    dt = params[0].dtype
            if dt is not None:
                dtype = dt
    if device is None:
        device = getattr(pipe, "_execution_device", "cpu")
    if dtype is None:
        dtype = torch_mod.float16
    return device, dtype


def build_conditioning(pipe, masked_image_pil, mask_pil):
    """Build 5-channel BrushNet conditioning latents matching vendor pipeline steps 4 and 6.1."""
    import torch
    import torch.nn.functional as F

    width, height = masked_image_pil.size
    device, dtype = _resolve_device_and_dtype(pipe, torch)
    mask_input = (
        mask_pil.convert("RGB")
        if isinstance(mask_pil, Image.Image) and mask_pil.mode != "RGB"
        else mask_pil
    )

    image = pipe.prepare_image(
        image=masked_image_pil,
        width=width,
        height=height,
        batch_size=1,
        num_images_per_prompt=1,
        device=device,
        dtype=dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    )
    original_mask = pipe.prepare_image(
        image=mask_input,
        width=width,
        height=height,
        batch_size=1,
        num_images_per_prompt=1,
        device=device,
        dtype=dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    )
    original_mask = (original_mask.sum(1)[:, None, :, :] < 0).to(image.dtype)

    with torch.no_grad():
        conditioning_latents = (
            pipe.vae.encode(
                image.to(device=device, dtype=dtype)
            ).latent_dist.sample()
            * pipe.vae.config.scaling_factor
        )
        mask_latents = F.interpolate(
            original_mask,
            size=(
                conditioning_latents.shape[-2],
                conditioning_latents.shape[-1],
            ),
        )
        conditioning_latents = torch.concat(
            [conditioning_latents, mask_latents], 1
        )
    return conditioning_latents


def precompute_prompt_embeds(pipe, device, torch_mod):
    """Precompute conditional BrushNet and UNet prompt embeddings, then offload text encoders to CPU."""
    pipe.text_encoder.to(device)
    pipe.text_encoder_brushnet.to(device)
    prompt_u = pp.RECIPES_PP["official"]["promptU"]

    with torch_mod.no_grad():
        brushnet_embeds = pipe._encode_prompt(
            pp.PROMPT_A,
            pp.PROMPT_B,
            pp.TRADOFF,
            device,
            1,
            False,
            negative_promptA=None,
            negative_promptB=None,
            t_nag=pp.TRADOFF_NAG,
        )
        try:
            raw_unet_embeds = pipe.encode_prompt(prompt_u, device, 1, False)
        except TypeError:
            raw_unet_embeds = pipe.encode_prompt(
                prompt_u, device, 1, True, negative_prompt=""
            )

        if isinstance(raw_unet_embeds, (tuple, list)):
            unet_embeds = raw_unet_embeds[0]
        elif (
            getattr(raw_unet_embeds, "ndim", 0) == 3
            and raw_unet_embeds.shape[0] == 2
        ):
            unet_embeds = raw_unet_embeds[1:2]
        elif getattr(raw_unet_embeds, "ndim", 0) == 2:
            unet_embeds = raw_unet_embeds.unsqueeze(0)
        else:
            unet_embeds = raw_unet_embeds

        brushnet_embeds = brushnet_embeds.detach().clone().to(device=device)
        unet_embeds = unet_embeds.detach().clone().to(device=device)

    pipe.text_encoder.to("cpu")
    pipe.text_encoder_brushnet.to("cpu")
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    return brushnet_embeds, unet_embeds


def set_submodule_gradient_checkpointing(module, enabled):
    """Set m.gradient_checkpointing = enabled on every submodule that defines the attribute."""
    count = 0
    if hasattr(module, "modules"):
        for sub in module.modules():
            if hasattr(sub, "gradient_checkpointing"):
                sub.gradient_checkpointing = bool(enabled)
                # diffusers>=0.32 blocks inside the vendor models call this; old-API vendor never sets it
                if enabled and hasattr(sub, "_gradient_checkpointing_func"):
                    import functools

                    import torch.utils.checkpoint

                    sub._gradient_checkpointing_func = functools.partial(
                        torch.utils.checkpoint.checkpoint, use_reentrant=False
                    )
                count += 1
    return count


def save_lora_adapter(brushnet, lora_config, target_dir, torch_mod):
    """Save PEFT BrushNet LoRA state dict in fp16 safetensors and adapter_config.json."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    raw_state = get_peft_model_state_dict(brushnet)
    fp16_state = {
        k: v.detach().to(dtype=torch_mod.float16).contiguous().cpu()
        for k, v in raw_state.items()
    }
    weights_path = target_path / "brushnet_lora.safetensors"
    save_file(fp16_state, str(weights_path))

    config_path = target_path / "adapter_config.json"
    if hasattr(lora_config, "save_pretrained"):
        lora_config.save_pretrained(str(target_path))
    if not config_path.is_file():
        if hasattr(lora_config, "to_dict"):
            cfg_dict = lora_config.to_dict()
        else:
            cfg_dict = {
                "peft_type": "LORA",
                "r": getattr(lora_config, "r", 8),
                "lora_alpha": getattr(lora_config, "lora_alpha", 8),
                "init_lora_weights": getattr(
                    lora_config, "init_lora_weights", "gaussian"
                ),
                "target_modules": list(
                    getattr(
                        lora_config, "target_modules", LORA_TARGET_MODULES
                    )
                ),
            }
        for key, val in list(cfg_dict.items()):
            if isinstance(val, set):
                cfg_dict[key] = sorted(val)
        config_path.write_text(
            json.dumps(cfg_dict, indent=2), encoding="utf-8"
        )

    return sha256_file(weights_path)


def _positive_int(value):
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return ival


def _positive_float(value):
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive float")
    return fval


def _validate_run_name_arg(value):
    if not isinstance(value, str) or not RUN_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("Invalid run name")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_name",
        type=_validate_run_name_arg,
        help="Fresh LoRA run name under .local-artifacts/outpainting/lora-runs",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing training images (non-recursive)",
    )
    parser.add_argument(
        "--exclude-manifest",
        required=True,
        help="Path to frozen evaluation dataset manifest.json",
    )
    parser.add_argument(
        "--steps",
        type=_positive_int,
        default=600,
        help="Number of optimizer steps (default: 600)",
    )
    parser.add_argument(
        "--rank",
        type=_positive_int,
        default=8,
        help="LoRA rank and alpha (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=_positive_float,
        default=1e-4,
        help="AdamW learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--grad-accum",
        type=_positive_int,
        default=4,
        help="Gradient accumulation micro-steps per optimizer step (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--seg-model",
        default=None,
        help="anime-seg ISNet ONNX; when set, loss counts background pixels only",
    )
    parser.add_argument(
        "--bg-dir",
        default=None,
        help="Directory of character-free backgrounds; trains on characters composited "
        "onto them (needs --seg-model), full loss",
    )
    parser.add_argument(
        "--plain-prob",
        type=float,
        default=PLAIN_BACKGROUND_PROB,
        help="Share of composite samples using a synthetic plain gradient backdrop",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run 3 optimizer steps, print peak VRAM MB and s/step, and write only records.json",
    )
    args = parser.parse_args(argv)
    if args.bg_dir and not args.seg_model:
        parser.error("--bg-dir requires --seg-model")
    return args


def main(argv=None):
    args = parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(LAB / "cache")

    collected, exclude_dataset_id = collect_training_images(
        args.images_dir, args.exclude_manifest
    )
    train_sha256_list = [sha for _, sha in collected]

    loaded_images = []
    for img_path, _ in collected:
        with Image.open(img_path) as raw_img:
            loaded_images.append(
                ImageOps.exif_transpose(raw_img).convert("RGB")
            )

    char_masks = None
    seg_sha256 = None
    char_fraction = None
    if args.seg_model:
        import onnxruntime as ort

        seg_sha256 = sha256_file(Path(args.seg_model))
        session = ort.InferenceSession(
            str(args.seg_model), providers=["CPUExecutionProvider"]
        )
        char_masks = [compute_character_mask(img, session) for img in loaded_images]
        del session
        char_fraction = round(
            sum(sum(i * c for i, c in enumerate(m.histogram())) / (255 * m.width * m.height)
                for m in char_masks) / len(char_masks),
            4,
        )

    bg_paths = None
    if args.bg_dir:
        bg_paths = sorted(
            p
            for p in Path(args.bg_dir).iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        if len(bg_paths) < 100:
            raise SystemExit("Need at least 100 background images")

    model_info = pp.verify_powerpaint_preflight(LAB, base="realistic")
    run_dir = create_lora_run_dir(LAB, args.run_name)

    import diffusers
    from diffusers import DDPMScheduler
    import peft
    from peft import LoraConfig, inject_adapter_in_model
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run BrushNet LoRA training")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch_gen = torch.Generator(device="cuda").manual_seed(args.seed)

    device = torch.device("cuda")
    weight_dtype = torch.float16
    effective_steps = PROBE_STEPS if args.probe else args.steps

    with single_instance():
        gate = SimpleNamespace(LAB=pp.LAB, torch=torch)
        pipe = None
        try:
            torch.set_grad_enabled(True)
            pipe = pp.load_powerpaint_pipeline(
                gate, model_info=model_info, base="realistic", offload=False
            )

            for module in (
                pipe.vae,
                pipe.unet,
                pipe.brushnet,
                pipe.text_encoder,
                pipe.text_encoder_brushnet,
            ):
                module.requires_grad_(False)

            brushnet_embeds, unet_embeds = precompute_prompt_embeds(
                pipe, device, torch
            )

            pipe.vae.to(device=device, dtype=weight_dtype)
            pipe.unet.to(device=device, dtype=weight_dtype)
            pipe.brushnet.to(device=device, dtype=weight_dtype)

            lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=list(LORA_TARGET_MODULES),
            )
            inject_adapter_in_model(lora_config, pipe.brushnet)

            lora_params = []
            for param in pipe.brushnet.parameters():
                if param.requires_grad:
                    param.data = param.data.to(
                        device=device, dtype=torch.float32
                    )
                    lora_params.append(param)
            if not lora_params:
                raise RuntimeError("No trainable LoRA parameters found in brushnet")

            optimizer = torch.optim.AdamW(
                lora_params, lr=args.lr, weight_decay=1e-2
            )
            noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            scaler = torch.amp.GradScaler("cuda")

            set_submodule_gradient_checkpointing(pipe.unet, True)
            set_submodule_gradient_checkpointing(pipe.brushnet, True)
            grad_ckpt_enabled = True
            grad_ckpt_error = None

            pipe.vae.eval()
            pipe.unet.train()
            pipe.brushnet.train()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = perf_counter()

            loss_history = []
            optimizer.zero_grad(set_to_none=True)

            def _run_micro_step(target_tensor, cond_latents, noise, timesteps, weight=None):
                noisy = noise_scheduler.add_noise(
                    target_tensor, noise, timesteps
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    down, mid, up = pipe.brushnet(
                        noisy,
                        timesteps,
                        encoder_hidden_states=brushnet_embeds,
                        brushnet_cond=cond_latents,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )
                    pred = pipe.unet(
                        noisy,
                        timesteps,
                        encoder_hidden_states=unet_embeds,
                        down_block_add_samples=down,
                        mid_block_add_sample=mid,
                        up_block_add_samples=up,
                        return_dict=False,
                    )[0]
                    if weight is None:
                        loss_val = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                    else:
                        sq = (pred.float() - noise.float()) ** 2 * weight
                        loss_val = sq.sum() / (weight.sum() * sq.shape[1]).clamp_min(1.0)
                    loss_val = loss_val / args.grad_accum
                scaler.scale(loss_val).backward()
                return float(loss_val.detach().item())

            for step in range(1, effective_steps + 1):
                step_loss = 0.0
                for _ in range(args.grad_accum):
                    img_idx = rng.randrange(len(loaded_images))
                    if bg_paths:
                        background = None
                        if rng.random() >= args.plain_prob:
                            with Image.open(rng.choice(bg_paths)) as bg_file:
                                background = bg_file.convert("RGB")
                        sample = generate_composite_example(
                            loaded_images[img_idx], char_masks[img_idx], background, rng
                        )
                    else:
                        sample = generate_training_example(
                            loaded_images[img_idx],
                            rng,
                            char_mask=char_masks[img_idx] if char_masks else None,
                        )
                    weight = None
                    if sample["loss_weight"] is not None:
                        weight = torch.from_numpy(
                            latent_loss_weight(sample["loss_weight"])
                        ).to(device)[None, None]

                    with torch.no_grad():
                        target_img_tensor = pipe.prepare_image(
                            image=sample["target"],
                            width=pp.INFERENCE_WIDTH,
                            height=pp.INFERENCE_HEIGHT,
                            batch_size=1,
                            num_images_per_prompt=1,
                            device=device,
                            dtype=weight_dtype,
                            do_classifier_free_guidance=False,
                            guess_mode=False,
                        )
                        latents = (
                            pipe.vae.encode(
                                target_img_tensor.to(
                                    device=device, dtype=weight_dtype
                                )
                            ).latent_dist.sample()
                            * pipe.vae.config.scaling_factor
                        )
                        cond = build_conditioning(
                            pipe, sample["masked_input"], sample["mask"]
                        )
                        noise = torch.randn(
                            latents.shape,
                            generator=torch_gen,
                            device=device,
                            dtype=latents.dtype,
                        )
                        t = torch.randint(
                            0,
                            int(noise_scheduler.config.num_train_timesteps),
                            (1,),
                            generator=torch_gen,
                            device=device,
                        ).long()

                    if grad_ckpt_enabled:
                        try:
                            micro_loss = _run_micro_step(latents, cond, noise, t, weight)
                            if any(p.grad is None for p in lora_params):
                                raise RuntimeError(
                                    "Gradient checkpointing produced None gradients"
                                )
                        except Exception as ckpt_error:
                            optimizer.zero_grad(set_to_none=True)
                            set_submodule_gradient_checkpointing(
                                pipe.unet, False
                            )
                            set_submodule_gradient_checkpointing(
                                pipe.brushnet, False
                            )
                            grad_ckpt_enabled = False
                            grad_ckpt_error = f"{type(ckpt_error).__name__}: {ckpt_error}"[:500]
                            micro_loss = _run_micro_step(latents, cond, noise, t, weight)
                    else:
                        micro_loss = _run_micro_step(latents, cond, noise, t, weight)

                    step_loss += micro_loss

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if (
                    args.probe
                    or step % LOSS_LOG_INTERVAL == 0
                    or step == 1
                    or step == effective_steps
                ):
                    loss_history.append(
                        {"step": step, "loss": round(step_loss, 6)}
                    )
                    print(
                        f"step={step}/{effective_steps} loss={step_loss:.6f}",
                        flush=True,
                    )

                if not args.probe and step % CHECKPOINT_INTERVAL == 0:
                    ckpt_dir = run_dir / f"step-{step}"
                    save_lora_adapter(
                        pipe.brushnet, lora_config, ckpt_dir, torch
                    )

            torch.cuda.synchronize()
            elapsed_seconds = perf_counter() - started
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1048576
            seconds_per_step = elapsed_seconds / max(1, effective_steps)

            lora_sha256 = None
            if not args.probe:
                lora_sha256 = save_lora_adapter(
                    pipe.brushnet, lora_config, run_dir, torch
                )

            records = {
                "run_name": args.run_name,
                "probe": bool(args.probe),
                "steps": effective_steps,
                "requested_steps": args.steps,
                "rank": args.rank,
                "lr": args.lr,
                "grad_accum": args.grad_accum,
                "seed": args.seed,
                "train_image_count": len(train_sha256_list),
                "train_image_sha256": train_sha256_list,
                "exclude_manifest_dataset": exclude_dataset_id,
                "exclude_dataset_id": exclude_dataset_id,
                "loss_history": loss_history,
                "losses": loss_history,
                "peak_vram_mb": round(peak_vram_mb, 2),
                "seconds": round(elapsed_seconds, 4),
                "seconds_per_step": round(seconds_per_step, 4),
                "gradient_checkpointing": bool(grad_ckpt_enabled),
                "gradient_checkpointing_error": grad_ckpt_error,
                "torch_version": torch.__version__,
                "diffusers_version": diffusers.__version__,
                "peft_version": peft.__version__,
                "brushnet_lora_sha256": lora_sha256,
                "loss_mode": "composite" if bg_paths else ("background_only" if char_masks else "full"),
                "background_count": len(bg_paths) if bg_paths else None,
                "plain_background_prob": args.plain_prob if bg_paths else None,
                "seg_model": "skytnt/anime-seg@493cb608 isnetis.onnx" if char_masks else None,
                "seg_model_sha256": seg_sha256,
                "mean_character_fraction": char_fraction,
            }
            (run_dir / "records.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            print(
                f"peak_vram_mb={peak_vram_mb:.2f} s_per_step={seconds_per_step:.4f} steps={effective_steps} seconds={elapsed_seconds:.2f}",
                flush=True,
            )
            return records
        finally:
            pp.release_pipeline(pipe, gate)
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
