"""Isolated PowerPaint v2 (BrushNet) anime outpainting lab runner.

CLI:
  python .local-artifacts/outpainting/powerpaint_gpu.py RUN_NAME [--full]

Default smoke runs eval-b bottom cut seed 42 (1 output). Passing --full runs
both frozen evaluation fixtures (eval-a, eval-b) x 4 cut directions x 2 seeds
= 16 outputs. Lab evaluation only; never used in production.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageChops, ImageFilter, ImageOps

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from conditioning_gpu import (  # noqa: E402
    RUN_NAME_RE,
    direction_crop_and_margins,
    evaluation_sources,
    new_output,
    seam_crop_boxes,
    sha256_file,
)

SOFT = "anime illustration, continue the same existing character beyond the cropped edge, coherent anatomy, matching hair, clothing, line art, colors and lighting, continuation of the existing scene"
NEGATIVE = "duplicate character, extra face, extra eyes, extra limbs, malformed anatomy, text, watermark, frame, border, blurry, artifacts"

RECIPE = "powerpaint-v2"
POWERPAINT_DIR_NAME = "powerpaint-v2"
VENDOR_CODE_REVISION = "5b4c3d52291709fcec2a1870d987da693fd3549c"
POWERPAINT_REPOSITORY = "JunhaoZhuang/PowerPaint_v2"
POWERPAINT_MODEL_REVISION = "49ce8271db56ea91e7e9412789ba7faa2e7c83f1"
EXPECTED_WEIGHT_FILE_COUNT = 15

BASE_MODEL_PREFIX = "realisticVisionV60B1_v51VAE"
BRUSHNET_PREFIX = "PowerPaint_Brushnet"
NONSTANDARD_UNET_FILENAME = (
    f"{BASE_MODEL_PREFIX}/unet/diffusion_pytorch_model-002.safetensors"
)
REALISTIC_TEXT_ENCODER_FILENAME = (
    f"{BASE_MODEL_PREFIX}/text_encoder/model.safetensors"
)
REALISTIC_VAE_FILENAME = (
    f"{BASE_MODEL_PREFIX}/vae/diffusion_pytorch_model.safetensors"
)
BRUSHNET_SAFETENSORS_FILENAME = (
    f"{BRUSHNET_PREFIX}/diffusion_pytorch_model.safetensors"
)
TASK_TEXT_ENCODER_BIN_FILENAME = f"{BRUSHNET_PREFIX}/pytorch_model.bin"

ANYLORA_REPOSITORY = "Lykon/AnyLoRA"
ANYLORA_REVISION = "5dac4354b7f5d32f140e1f0fbc849e92b744f347"
ANYLORA_UNET_REL_PATH = "unet/diffusion_pytorch_model.fp16.safetensors"
ANYLORA_TEXT_ENCODER_REL_PATH = "text_encoder/model.fp16.safetensors"
ANYLORA_VAE_REL_PATH = "vae/diffusion_pytorch_model.fp16.safetensors"
ANYLORA_REQUIRED_REL_FILES = (
    "unet/config.json",
    ANYLORA_UNET_REL_PATH,
    "text_encoder/config.json",
    ANYLORA_TEXT_ENCODER_REL_PATH,
    "vae/config.json",
    ANYLORA_VAE_REL_PATH,
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)

_REALISTIC_BASE_DIR = LAB / POWERPAINT_DIR_NAME / "weights" / BASE_MODEL_PREFIX
_ANYLORA_BASE_DIR = LAB / "base-models" / "anylora"

BASES = {
    "realistic": {
        "dir": _REALISTIC_BASE_DIR,
        "repository": POWERPAINT_REPOSITORY,
        "revision": POWERPAINT_MODEL_REVISION,
        "unet_weights_path": LAB / POWERPAINT_DIR_NAME / "weights" / NONSTANDARD_UNET_FILENAME,
        "unet_weights": LAB / POWERPAINT_DIR_NAME / "weights" / NONSTANDARD_UNET_FILENAME,
        "text_encoder_weights_path": LAB / POWERPAINT_DIR_NAME / "weights" / REALISTIC_TEXT_ENCODER_FILENAME,
        "text_encoder_weights": LAB / POWERPAINT_DIR_NAME / "weights" / REALISTIC_TEXT_ENCODER_FILENAME,
        "vae_dir": _REALISTIC_BASE_DIR / "vae",
        "vae_weights_path": LAB / POWERPAINT_DIR_NAME / "weights" / REALISTIC_VAE_FILENAME,
        "vae_weights": LAB / POWERPAINT_DIR_NAME / "weights" / REALISTIC_VAE_FILENAME,
        "variant": None,
    },
    "anylora": {
        "dir": _ANYLORA_BASE_DIR,
        "repository": ANYLORA_REPOSITORY,
        "revision": ANYLORA_REVISION,
        "unet_weights_path": _ANYLORA_BASE_DIR / ANYLORA_UNET_REL_PATH,
        "unet_weights": _ANYLORA_BASE_DIR / ANYLORA_UNET_REL_PATH,
        "text_encoder_weights_path": _ANYLORA_BASE_DIR / ANYLORA_TEXT_ENCODER_REL_PATH,
        "text_encoder_weights": _ANYLORA_BASE_DIR / ANYLORA_TEXT_ENCODER_REL_PATH,
        "vae_dir": _ANYLORA_BASE_DIR / "vae",
        "vae_weights_path": _ANYLORA_BASE_DIR / ANYLORA_VAE_REL_PATH,
        "vae_weights": _ANYLORA_BASE_DIR / ANYLORA_VAE_REL_PATH,
        "variant": "fp16",
    },
}

MANDATORY_WEIGHT_FILES = {
    BRUSHNET_SAFETENSORS_FILENAME: (
        3544366408,
        "745a3babce414c8b765c57e86412544cecdbdb0601648900d10b482256babb76",
    ),
    TASK_TEXT_ENCODER_BIN_FILENAME: (
        492401329,
        "15d6a233b5707d13fc7b47d763812a375fa171160e6544041fee9ee4caea6e4d",
    ),
    NONSTANDARD_UNET_FILENAME: (
        3438167536,
        "c921f91410f538194cd53d47907c9421c22af389113a160cc2643d1c39c4e1d4",
    ),
    REALISTIC_TEXT_ENCODER_FILENAME: (
        492265168,
        "c3f93620b7278480aeae09978486b94fdbfee2b0677dbb6f1744eef2d36f6cfc",
    ),
    REALISTIC_VAE_FILENAME: (
        334643268,
        "fc436097e5cbb105c44df4403ae84acf7e81de1f64acfedc334888e9a8eea223",
    ),
}

PLACEHOLDER_TOKENS = ("P_ctxt", "P_shape", "P_obj")
INITIALIZE_TOKENS = ("a", "a", "a")
NUM_VECTORS_PER_TOKEN = 10
REQUIRED_TASK_EMBEDDING_KEYS = tuple(
    f"text_model.embeddings.token_embedding.trainable_embeddings.{token}"
    for token in PLACEHOLDER_TOKENS
)
LEGACY_POSITION_IDS_KEY = "text_model.embeddings.position_ids"
DEFAULT_CLIP_MAX_POSITION_EMBEDDINGS = 77
BENIGN_MISSING_TEXT_ENCODER_KEYS = frozenset(
    {LEGACY_POSITION_IDS_KEY}
)

PROMPT_A = " P_ctxt"
PROMPT_B = " P_ctxt"
NEGATIVE_PROMPT_A = " P_obj"
NEGATIVE_PROMPT_B = " P_obj"
TRADOFF = 1.0
TRADOFF_NAG = 1.0
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.0
BRUSHNET_CONDITIONING_SCALE = 1.0
INFERENCE_WIDTH = 768
INFERENCE_HEIGHT = 432
OVERLAP_PX = 8
RECIPES_PP = {
    "lab": {
        "recipe_name": "lab",
        "promptU": SOFT,
        "negative_promptU": NEGATIVE,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "mask_overlap_px": 0,
    },
    "official": {
        "recipe_name": "official",
        "promptU": f"{SOFT}, empty scene",
        "negative_promptU": NEGATIVE,
        "guidance_scale": 10.0,
        "num_inference_steps": 45,
        "mask_overlap_px": OVERLAP_PX,
    },
}
MODEL_CPU_OFFLOAD_SEQ = (
    "text_encoder->text_encoder_brushnet->image_encoder->brushnet->unet->vae"
)


def expand_mask(mask, px):
    """Expand 255 outpainting region into 0 foreground region by px pixels using MaxFilter."""
    if not isinstance(mask, Image.Image) or mask.mode != "L":
        raise ValueError("Mask expansion requires an L-mode mask")
    if type(px) is not int or px < 0:
        raise ValueError("Overlap px must be a non-negative integer")
    if px == 0:
        return mask.copy()
    return mask.filter(ImageFilter.MaxFilter(2 * px + 1))


def prepare_brushnet_input(canvas, mask):
    """Set unknown masked pixels (mask=255) to black while keeping known pixels intact."""
    if not isinstance(canvas, Image.Image) or canvas.mode != "RGB":
        raise ValueError("BrushNet input preparation requires an RGB canvas")
    if not isinstance(mask, Image.Image) or mask.mode != "L":
        raise ValueError("BrushNet input preparation requires an L-mode mask")
    if canvas.size != mask.size:
        raise ValueError("Canvas and mask dimensions must match")
    black = Image.new("RGB", canvas.size, (0, 0, 0))
    return Image.composite(black, canvas, mask)


def _validate_relative_subpath(root_dir, resolved_root, raw_rel):
    """Validate that raw_rel is a safe relative path inside root_dir with no symlinks."""
    if not isinstance(raw_rel, str) or not raw_rel.strip():
        raise ValueError("Invalid empty relative path in manifest")
    if "\\" in raw_rel:
        raise ValueError(f"Backslash not allowed in manifest path: {raw_rel}")
    rel_obj = Path(raw_rel)
    if rel_obj.is_absolute() or ".." in rel_obj.parts or not rel_obj.parts:
        raise ValueError(f"Manifest path escapes weights directory: {raw_rel}")
    candidate = root_dir
    for part in rel_obj.parts:
        if part in ("", "."):
            raise ValueError(f"Unsafe path segment in manifest: {raw_rel}")
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink not allowed in weights tree: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root) or resolved == resolved_root:
        raise ValueError(f"Manifest path escapes weights directory: {raw_rel}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing weights file: {candidate}")
    return rel_obj.as_posix(), resolved


def _stream_sha256(path):
    """Compute SHA-256 of a file using 1 MiB streaming chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_anylora_preflight(lab_root):
    """Verify AnyLoRA SD1.5 base model files, UNet config, and streaming weight hashes."""
    lab_path = Path(lab_root)
    if lab_path.is_symlink() or not lab_path.is_dir():
        raise FileNotFoundError(f"Invalid lab root directory: {lab_path}")
    resolved_lab = lab_path.resolve()

    base_models_dir = lab_path / "base-models"
    if base_models_dir.is_symlink():
        raise ValueError(
            f"Symlink not allowed in weights tree: {base_models_dir}"
        )
    anylora_dir = base_models_dir / "anylora"
    if anylora_dir.is_symlink():
        raise ValueError(f"Symlink not allowed in weights tree: {anylora_dir}")
    if not base_models_dir.is_dir() or not anylora_dir.is_dir():
        raise FileNotFoundError(
            f"Missing AnyLoRA base model directory: {anylora_dir}"
        )
    resolved_anylora_dir = anylora_dir.resolve()
    if (
        not resolved_anylora_dir.is_relative_to(resolved_lab)
        or resolved_anylora_dir == resolved_lab
    ):
        raise ValueError("AnyLoRA base model directory escapes lab root")

    resolved_files = {}
    for rel_raw in ANYLORA_REQUIRED_REL_FILES:
        rel_posix, resolved_file = _validate_relative_subpath(
            anylora_dir, resolved_anylora_dir, rel_raw
        )
        if not resolved_file.is_relative_to(resolved_lab):
            raise ValueError(
                f"AnyLoRA file escapes lab directory: {resolved_file}"
            )
        resolved_files[rel_posix] = resolved_file

    tokenizer_dir = anylora_dir / "tokenizer"
    for item in sorted(tokenizer_dir.iterdir()):
        if item.is_symlink():
            raise ValueError(f"Symlink not allowed in weights tree: {item}")
        resolved_item = item.resolve()
        if (
            not resolved_item.is_relative_to(resolved_anylora_dir)
            or not resolved_item.is_relative_to(resolved_lab)
            or not resolved_item.is_file()
        ):
            raise ValueError(f"Invalid tokenizer entry in AnyLoRA: {item}")

    unet_config_path = resolved_files["unet/config.json"]
    unet_config = json.loads(unet_config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(unet_config, dict):
        raise ValueError("Invalid AnyLoRA UNet config JSON")
    if unet_config.get("in_channels") != 4:
        raise ValueError(
            f"AnyLoRA UNet config must have in_channels == 4, got {unet_config.get('in_channels')!r}"
        )
    if unet_config.get("cross_attention_dim") != 768:
        raise ValueError(
            f"AnyLoRA UNet config must have cross_attention_dim == 768, got {unet_config.get('cross_attention_dim')!r}"
        )

    for cfg_rel in (
        "text_encoder/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    ):
        cfg_obj = json.loads(
            resolved_files[cfg_rel].read_text(encoding="utf-8-sig")
        )
        if not isinstance(cfg_obj, dict):
            raise ValueError(f"Invalid AnyLoRA config JSON: {cfg_rel}")

    scheduler_config_path = resolved_files["scheduler/scheduler_config.json"]
    scheduler_config = json.loads(
        scheduler_config_path.read_text(encoding="utf-8-sig")
    )

    unet_weights_path = resolved_files[ANYLORA_UNET_REL_PATH]
    text_encoder_weights_path = resolved_files[ANYLORA_TEXT_ENCODER_REL_PATH]
    vae_weights_path = resolved_files[ANYLORA_VAE_REL_PATH]

    base_weights_sha256 = {
        "unet": _stream_sha256(unet_weights_path),
        "text_encoder": _stream_sha256(text_encoder_weights_path),
        "vae": _stream_sha256(vae_weights_path),
    }

    pp_weights_dir = lab_path / POWERPAINT_DIR_NAME / "weights"
    realistic_base_dir = pp_weights_dir / BASE_MODEL_PREFIX
    return {
        "repository": POWERPAINT_REPOSITORY,
        "revision": POWERPAINT_MODEL_REVISION,
        "code_revision": VENDOR_CODE_REVISION,
        "base_name": "anylora",
        "base_repository": ANYLORA_REPOSITORY,
        "base_revision": ANYLORA_REVISION,
        "base_variant": "fp16",
        "weights_dir": pp_weights_dir,
        "base_model_dir": anylora_dir,
        "realistic_base_dir": realistic_base_dir,
        "brushnet_dir": pp_weights_dir / BRUSHNET_PREFIX,
        "unet_weights_path": unet_weights_path,
        "text_encoder_weights_path": text_encoder_weights_path,
        "vae_dir": anylora_dir / "vae",
        "vae_weights_path": vae_weights_path,
        "base_weights_sha256": base_weights_sha256,
        "hashes": dict(base_weights_sha256),
        "manifest_sha256": None,
        "unet_config": unet_config,
        "scheduler_config": scheduler_config,
        "scheduler_config_path": scheduler_config_path.relative_to(
            resolved_lab
        ).as_posix(),
        "scheduler_config_sha256": _stream_sha256(scheduler_config_path),
    }


def _verify_realistic_powerpaint_preflight(lab_root):
    """Verify PowerPaint v2 weights manifest, 15 files, and pinned hashes before model imports."""
    lab_path = Path(lab_root)
    pp_root = lab_path / POWERPAINT_DIR_NAME
    weights_dir = pp_root / "weights"
    if weights_dir.is_symlink() or not weights_dir.is_dir():
        raise FileNotFoundError(f"Missing PowerPaint weights directory: {weights_dir}")
    resolved_weights_dir = weights_dir.resolve()

    manifest_path = weights_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing PowerPaint weights manifest: {manifest_path}")
    resolved_manifest = manifest_path.resolve()
    if (
        not resolved_manifest.is_relative_to(resolved_weights_dir)
        or resolved_manifest.parent != resolved_weights_dir
    ):
        raise ValueError("PowerPaint manifest escapes weights directory")

    manifest_sha256 = sha256_file(resolved_manifest)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("PowerPaint manifest must be a JSON object")
    if manifest.get("complete") is not True:
        raise ValueError("PowerPaint weights manifest is not marked complete=true")

    manifest_repo = (
        manifest.get("repo")
        or manifest.get("repository")
        or manifest.get("repo_id")
        or manifest.get("model_id")
    )
    if manifest_repo != POWERPAINT_REPOSITORY:
        raise ValueError(
            f"Unexpected PowerPaint repository: {manifest_repo}"
        )
    if manifest.get("revision") != POWERPAINT_MODEL_REVISION:
        raise ValueError(
            f"Unexpected PowerPaint revision: {manifest.get('revision')}"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_WEIGHT_FILE_COUNT:
        raise ValueError(
            f"PowerPaint manifest must list exactly {EXPECTED_WEIGHT_FILE_COUNT} files"
        )

    verified_hashes = {}
    verified_sizes = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Manifest file entry must be a JSON object")
        rel_raw = entry.get("file", entry.get("path"))
        rel_posix, resolved_file = _validate_relative_subpath(
            weights_dir, resolved_weights_dir, rel_raw
        )
        if rel_posix in verified_hashes:
            raise ValueError(f"Duplicate file entry in PowerPaint manifest: {rel_posix}")

        if "bytes" not in entry and "size" not in entry:
            raise ValueError(f"Missing size for {rel_posix} in PowerPaint manifest")
        expected_bytes = int(
            entry["bytes"] if "bytes" in entry else entry["size"]
        )
        expected_sha256 = str(entry.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Invalid sha256 in PowerPaint manifest for {rel_posix}")

        actual_bytes = resolved_file.stat().st_size
        assert (
            actual_bytes == expected_bytes
        ), f"Size mismatch for {rel_posix}: {actual_bytes} != {expected_bytes}"
        actual_sha256 = sha256_file(resolved_file)
        assert (
            actual_sha256 == expected_sha256
        ), f"SHA-256 mismatch vs manifest for {rel_posix}"

        if rel_posix in MANDATORY_WEIGHT_FILES:
            pinned_bytes, pinned_sha256 = MANDATORY_WEIGHT_FILES[rel_posix]
            assert (
                actual_bytes == pinned_bytes
            ), f"Pinned size mismatch for {rel_posix}: {actual_bytes} != {pinned_bytes}"
            assert (
                actual_sha256 == pinned_sha256
            ), f"Pinned SHA-256 mismatch for {rel_posix}"

        verified_hashes[rel_posix] = actual_sha256
        verified_sizes[rel_posix] = actual_bytes

    for mandatory_rel, (pinned_bytes, pinned_sha256) in MANDATORY_WEIGHT_FILES.items():
        if mandatory_rel not in verified_hashes:
            raise ValueError(
                f"Missing mandatory PowerPaint weight entry: {mandatory_rel}"
            )
        assert verified_sizes[mandatory_rel] == pinned_bytes
        assert verified_hashes[mandatory_rel] == pinned_sha256

    base_dir = weights_dir / BASE_MODEL_PREFIX
    unet_config_path = base_dir / "unet" / "config.json"
    scheduler_config_path = base_dir / "scheduler" / "scheduler_config.json"
    if not unet_config_path.is_file():
        raise FileNotFoundError(f"Missing local UNet config: {unet_config_path}")
    if not scheduler_config_path.is_file():
        raise FileNotFoundError(
            f"Missing local scheduler config: {scheduler_config_path}"
        )
    unet_config = json.loads(unet_config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(unet_config, dict):
        raise ValueError("Invalid local UNet config JSON")
    scheduler_config = json.loads(
        scheduler_config_path.read_text(encoding="utf-8-sig")
    )
    if not isinstance(scheduler_config, dict):
        raise ValueError("Invalid local scheduler config JSON")

    base_weights_sha256 = {
        "unet": verified_hashes[NONSTANDARD_UNET_FILENAME],
        "text_encoder": verified_hashes[REALISTIC_TEXT_ENCODER_FILENAME],
        "vae": verified_hashes[REALISTIC_VAE_FILENAME],
    }

    return {
        "repository": POWERPAINT_REPOSITORY,
        "revision": POWERPAINT_MODEL_REVISION,
        "code_revision": VENDOR_CODE_REVISION,
        "base_name": "realistic",
        "base_repository": BASES["realistic"]["repository"],
        "base_revision": BASES["realistic"]["revision"],
        "base_variant": BASES["realistic"]["variant"],
        "weights_dir": weights_dir,
        "base_model_dir": base_dir,
        "realistic_base_dir": base_dir,
        "brushnet_dir": weights_dir / BRUSHNET_PREFIX,
        "unet_weights_path": weights_dir / NONSTANDARD_UNET_FILENAME,
        "text_encoder_weights_path": weights_dir / REALISTIC_TEXT_ENCODER_FILENAME,
        "vae_dir": base_dir / "vae",
        "vae_weights_path": weights_dir / REALISTIC_VAE_FILENAME,
        "base_weights_sha256": base_weights_sha256,
        "manifest_sha256": manifest_sha256,
        "hashes": verified_hashes,
        "unet_config": unet_config,
        "scheduler_config": scheduler_config,
        "scheduler_config_path": scheduler_config_path.relative_to(lab_path).as_posix(),
        "scheduler_config_sha256": sha256_file(scheduler_config_path),
    }


def verify_powerpaint_preflight(lab_root, base="realistic"):
    """Verify PowerPaint v2 and selected SD1.5 base model preflight checks."""
    if not isinstance(base, str) or base not in BASES:
        raise ValueError(f"Unsupported base model: {base!r}")
    if base == "realistic":
        return _verify_realistic_powerpaint_preflight(lab_root)

    anylora_info = verify_anylora_preflight(lab_root)
    lab_path = Path(lab_root)
    brushnet_path = (
        lab_path / POWERPAINT_DIR_NAME / "weights" / BRUSHNET_SAFETENSORS_FILENAME
    )
    pinned_brushnet_bytes = MANDATORY_WEIGHT_FILES[BRUSHNET_SAFETENSORS_FILENAME][0]
    if (
        brushnet_path.is_file()
        and not brushnet_path.is_symlink()
        and brushnet_path.stat().st_size == pinned_brushnet_bytes
    ):
        pp_info = _verify_realistic_powerpaint_preflight(lab_root)
        pp_info.update(
            {
                "base_name": anylora_info["base_name"],
                "base_repository": anylora_info["base_repository"],
                "base_revision": anylora_info["base_revision"],
                "base_variant": anylora_info["base_variant"],
                "base_model_dir": anylora_info["base_model_dir"],
                "unet_config": anylora_info["unet_config"],
                "unet_weights_path": anylora_info["unet_weights_path"],
                "text_encoder_weights_path": anylora_info[
                    "text_encoder_weights_path"
                ],
                "vae_dir": anylora_info["vae_dir"],
                "vae_weights_path": anylora_info["vae_weights_path"],
                "base_weights_sha256": anylora_info["base_weights_sha256"],
            }
        )
        return pp_info
    return anylora_info


def _extract_flat_position_values(tensor):
    """Extract a flat list of scalar values from a real or fake position_ids tensor."""
    return tensor.detach().cpu().reshape(-1).tolist()


def _validate_legacy_position_ids_buffer(
    text_encoder_brushnet, pos_tensor, torch_mod
):
    """Verify that checkpoint position_ids is the exact canonical 0..N-1 integer buffer."""
    if not isinstance(pos_tensor, torch_mod.Tensor):
        raise ValueError(
            f"Non-tensor entry for {LEGACY_POSITION_IDS_KEY} in task checkpoint"
        )
    if (
        hasattr(pos_tensor, "is_floating_point")
        and callable(pos_tensor.is_floating_point)
        and pos_tensor.is_floating_point()
    ):
        raise ValueError(
            f"{LEGACY_POSITION_IDS_KEY} buffer must have an integer dtype"
        )
    if (
        hasattr(pos_tensor, "is_complex")
        and callable(pos_tensor.is_complex)
        and pos_tensor.is_complex()
    ):
        raise ValueError(
            f"{LEGACY_POSITION_IDS_KEY} buffer must have an integer dtype"
        )
    dtype_str = str(getattr(pos_tensor, "dtype", "")).lower()
    if any(tag in dtype_str for tag in ("float", "fp", "complex", "bool")):
        raise ValueError(
            f"{LEGACY_POSITION_IDS_KEY} buffer must have an integer dtype, got {pos_tensor.dtype!r}"
        )

    embeddings = getattr(
        getattr(text_encoder_brushnet, "text_model", None), "embeddings", None
    )
    runtime_pos_ids = getattr(embeddings, "position_ids", None)
    actual_shape = tuple(getattr(pos_tensor, "shape", ()))

    if runtime_pos_ids is not None:
        if not isinstance(runtime_pos_ids, torch_mod.Tensor):
            raise ValueError("Runtime position_ids attribute is not a Tensor")
        expected_shape = tuple(runtime_pos_ids.shape)
        if not expected_shape or runtime_pos_ids.numel() <= 0:
            raise ValueError("Invalid runtime position_ids shape")
        max_positions = int(expected_shape[-1])
        if int(runtime_pos_ids.numel()) != max_positions or max_positions <= 0:
            raise ValueError(
                f"Invalid runtime position_ids dimensions: {expected_shape}"
            )
        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {LEGACY_POSITION_IDS_KEY}: {actual_shape} != {expected_shape}"
            )
    else:
        max_positions = None
        pos_embedding = getattr(embeddings, "position_embedding", None)
        pos_weight = getattr(pos_embedding, "weight", None)
        if pos_weight is not None and getattr(pos_weight, "shape", None):
            max_positions = int(pos_weight.shape[0])
        elif getattr(pos_embedding, "num_embeddings", None) is not None:
            max_positions = int(pos_embedding.num_embeddings)
        else:
            config = getattr(text_encoder_brushnet, "config", None) or getattr(
                getattr(text_encoder_brushnet, "text_model", None),
                "config",
                None,
            )
            if getattr(config, "max_position_embeddings", None) is not None:
                max_positions = int(config.max_position_embeddings)
        if max_positions is None:
            max_positions = DEFAULT_CLIP_MAX_POSITION_EMBEDDINGS
        if max_positions <= 0:
            raise ValueError(f"Invalid max_position_embeddings: {max_positions}")
        expected_shape = (1, max_positions)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {LEGACY_POSITION_IDS_KEY}: {actual_shape} != {expected_shape}"
            )

    if int(pos_tensor.numel()) != max_positions:
        raise ValueError(
            f"Unexpected numel for {LEGACY_POSITION_IDS_KEY}: {pos_tensor.numel()} != {max_positions}"
        )

    flat_values = _extract_flat_position_values(pos_tensor)
    if len(flat_values) != max_positions:
        raise ValueError(
            f"Unexpected value count for {LEGACY_POSITION_IDS_KEY}: {len(flat_values)} != {max_positions}"
        )
    if not all(
        isinstance(v, int) and not isinstance(v, bool) for v in flat_values
    ):
        raise ValueError(
            f"Non-integer values in {LEGACY_POSITION_IDS_KEY} buffer"
        )
    expected_values = list(range(max_positions))
    if flat_values != expected_values:
        raise ValueError(
            f"Unexpected positional IDs values in {LEGACY_POSITION_IDS_KEY}"
        )

    if runtime_pos_ids is not None:
        runtime_values = _extract_flat_position_values(runtime_pos_ids)
        if flat_values != runtime_values:
            raise ValueError(
                f"{LEGACY_POSITION_IDS_KEY} values do not match runtime position_ids buffer"
            )


def load_task_text_encoder_state(text_encoder_brushnet, state_dict, torch_mod):
    """Validate and load task text encoder state dict without leaving unverified embeddings."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Task text encoder checkpoint must be a non-empty state_dict")
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch_mod.Tensor):
            raise ValueError(f"Non-tensor entry in task text encoder checkpoint: {key!r}")
        if value.numel() == 0 or not bool(torch_mod.isfinite(value).all()):
            raise ValueError(f"Invalid tensor values in task checkpoint for key: {key}")

    for req_key in REQUIRED_TASK_EMBEDDING_KEYS:
        if req_key not in state_dict:
            raise ValueError(f"Missing required task token tensor in checkpoint: {req_key}")
        tensor = state_dict[req_key]
        if tuple(tensor.shape) != (NUM_VECTORS_PER_TOKEN, 768):
            raise ValueError(
                f"Unexpected shape for {req_key}: {tuple(tensor.shape)}"
            )

    emb_wrapper = text_encoder_brushnet.text_model.embeddings.token_embedding
    before_task = {
        token: emb_wrapper.trainable_embeddings[token].detach().cpu().clone()
        for token in PLACEHOLDER_TOKENS
    }

    expected_keys = set(text_encoder_brushnet.state_dict().keys())
    filtered_state_dict = dict(state_dict)
    if (
        LEGACY_POSITION_IDS_KEY in filtered_state_dict
        and LEGACY_POSITION_IDS_KEY not in expected_keys
    ):
        _validate_legacy_position_ids_buffer(
            text_encoder_brushnet,
            filtered_state_dict[LEGACY_POSITION_IDS_KEY],
            torch_mod,
        )
        filtered_state_dict.pop(LEGACY_POSITION_IDS_KEY)
    elif LEGACY_POSITION_IDS_KEY in filtered_state_dict:
        _validate_legacy_position_ids_buffer(
            text_encoder_brushnet,
            filtered_state_dict[LEGACY_POSITION_IDS_KEY],
            torch_mod,
        )
    elif LEGACY_POSITION_IDS_KEY in expected_keys:
        embeddings = getattr(
            getattr(text_encoder_brushnet, "text_model", None),
            "embeddings",
            None,
        )
        runtime_pos_ids = getattr(embeddings, "position_ids", None)
        _validate_legacy_position_ids_buffer(
            text_encoder_brushnet,
            runtime_pos_ids,
            torch_mod,
        )
        filtered_state_dict[LEGACY_POSITION_IDS_KEY] = runtime_pos_ids

    checkpoint_keys = set(filtered_state_dict.keys())
    missing_keys = sorted(expected_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - expected_keys)

    if missing_keys or unexpected_keys:
        print(
            f"Task text encoder state inspection: missing={missing_keys}, unexpected={unexpected_keys}",
            flush=True,
        )
        if unexpected_keys:
            raise ValueError(
                f"Unexpected keys in task text encoder checkpoint: {unexpected_keys}"
            )
        raise ValueError(
            f"Missing required keys in task text encoder checkpoint: {missing_keys}"
        )

    incompatible = text_encoder_brushnet.load_state_dict(
        filtered_state_dict, strict=True
    )
    if incompatible is not None:
        if list(getattr(incompatible, "unexpected_keys", ())):
            raise ValueError(
                f"Unexpected keys during task text encoder load: {incompatible.unexpected_keys}"
            )
        if list(getattr(incompatible, "missing_keys", ())):
            raise ValueError(
                f"Missing keys during task text encoder load: {incompatible.missing_keys}"
            )

    for token in PLACEHOLDER_TOKENS:
        key = f"text_model.embeddings.token_embedding.trainable_embeddings.{token}"
        loaded_param = emb_wrapper.trainable_embeddings[token]
        target_tensor = state_dict[key].to(
            device=loaded_param.device, dtype=loaded_param.dtype
        )
        if not torch_mod.allclose(loaded_param.detach(), target_tensor):
            raise ValueError(f"Task token embedding {token} did not match checkpoint")
        if torch_mod.allclose(
            loaded_param.detach().cpu().float(), before_task[token].float()
        ):
            raise ValueError(
                f"Task token embedding {token} remained equal to placeholder initialization"
            )

    for ext_entry in emb_wrapper.external_embeddings:
        name = ext_entry["name"]
        if name in emb_wrapper.trainable_embeddings:
            ext_entry["embedding"] = emb_wrapper.trainable_embeddings[name]


def _ensure_vendor_runtime_paths(lab_root):
    """Insert vendor and dependencies into sys.path at runtime only."""
    pp_root = Path(lab_root) / POWERPAINT_DIR_NAME
    vendor_dir = pp_root / "vendor"
    deps_dir = pp_root / "dependencies"
    if vendor_dir.is_symlink() or not vendor_dir.is_dir():
        raise FileNotFoundError(f"Missing PowerPaint vendor directory: {vendor_dir}")
    if deps_dir.is_symlink() or not deps_dir.is_dir():
        raise FileNotFoundError(
            f"Missing PowerPaint dependencies directory: {deps_dir}"
        )
    for runtime_dir in (vendor_dir, deps_dir):
        path_str = str(runtime_dir)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def verify_lora_preflight(lab_root, lora_name):
    """Validate a trained BrushNet LoRA run directory and return its metadata and SHA-256."""
    if not isinstance(lora_name, str) or not RUN_NAME_RE.fullmatch(lora_name):
        raise ValueError(f"Invalid LoRA run name: {lora_name!r}")
    lab_path = Path(lab_root)
    if lab_path.is_symlink() or not lab_path.is_dir():
        raise FileNotFoundError(f"Invalid lab root directory: {lab_path}")
    resolved_lab = lab_path.resolve()

    lora_runs_dir = lab_path / "lora-runs"
    if lora_runs_dir.is_symlink():
        raise ValueError(f"Symlink not allowed in weights tree: {lora_runs_dir}")
    if not lora_runs_dir.is_dir():
        raise FileNotFoundError(f"Missing LoRA runs directory: {lora_runs_dir}")
    resolved_lora_runs = lora_runs_dir.resolve()
    if (
        not resolved_lora_runs.is_relative_to(resolved_lab)
        or resolved_lora_runs == resolved_lab
    ):
        raise ValueError("LoRA runs directory escapes lab root")

    run_dir = lora_runs_dir / lora_name
    if run_dir.is_symlink():
        raise ValueError(f"Symlink not allowed in weights tree: {run_dir}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing LoRA run directory: {run_dir}")
    resolved_run_dir = run_dir.resolve()
    if (
        not resolved_run_dir.is_relative_to(resolved_lora_runs)
        or resolved_run_dir.parent != resolved_lora_runs
    ):
        raise ValueError(f"LoRA run directory escapes lora-runs: {run_dir}")

    _, weights_path = _validate_relative_subpath(
        run_dir, resolved_run_dir, "brushnet_lora.safetensors"
    )
    _, config_path = _validate_relative_subpath(
        run_dir, resolved_run_dir, "adapter_config.json"
    )
    if not weights_path.is_relative_to(resolved_lab) or not config_path.is_relative_to(
        resolved_lab
    ):
        raise ValueError("LoRA file escapes lab directory")

    adapter_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(adapter_config, dict):
        raise ValueError("Invalid LoRA adapter_config.json")

    lora_sha256 = sha256_file(weights_path)
    return {
        "lora_run": lora_name,
        "lora_sha256": lora_sha256,
        "run_dir": resolved_run_dir,
        "weights_path": weights_path,
        "config_path": config_path,
        "adapter_config": adapter_config,
    }


def load_brushnet_lora(brushnet, lab_root, lora_name, dtype=None, lora_info=None):
    """Inject and load a trained BrushNet LoRA into brushnet using PEFT without PeftModel."""
    if lora_info is None:
        lora_info = verify_lora_preflight(lab_root, lora_name)
    import inspect
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    raw_cfg = dict(lora_info["adapter_config"])
    sig = inspect.signature(LoraConfig)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    cfg_kwargs = (
        raw_cfg
        if has_var_kw
        else {k: v for k, v in raw_cfg.items() if k in sig.parameters}
    )
    if isinstance(cfg_kwargs.get("target_modules"), list):
        cfg_kwargs["target_modules"] = list(cfg_kwargs["target_modules"])
    lora_config = LoraConfig(**cfg_kwargs)
    inject_adapter_in_model(lora_config, brushnet)
    state_dict = load_file(str(lora_info["weights_path"]))
    incompatible = set_peft_model_state_dict(brushnet, state_dict)
    if incompatible is not None and list(getattr(incompatible, "unexpected_keys", ())):
        raise ValueError(
            f"Unexpected keys when loading BrushNet LoRA: {incompatible.unexpected_keys}"
        )
    if dtype is not None and hasattr(brushnet, "to"):
        brushnet.to(dtype=dtype)
    return lora_info


def load_powerpaint_pipeline(
    gate, model_info=None, base="realistic", offload=True, lora=None
):
    """Construct local PowerPaint v2 BrushNet pipeline without auto-downloads or safety_checker."""
    if model_info is None:
        model_info = verify_powerpaint_preflight(gate.LAB, base=base)
    _ensure_vendor_runtime_paths(gate.LAB)

    from diffusers import AutoencoderKL, UniPCMultistepScheduler
    from powerpaint.models.BrushNet_CA import BrushNetModel
    from powerpaint.models.unet_2d_condition import UNet2DConditionModel
    from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
        StableDiffusionPowerPaintBrushNetPipeline,
    )
    from powerpaint.utils.utils import TokenizerWrapper, add_tokens
    from safetensors.torch import load_model
    from transformers import CLIPTextModel

    weight_dtype = gate.torch.float16
    weights_dir = model_info["weights_dir"]
    base_dir = model_info["base_model_dir"]
    realistic_base_dir = model_info.get(
        "realistic_base_dir", weights_dir / BASE_MODEL_PREFIX
    )
    unet_weights_path = model_info.get(
        "unet_weights_path", weights_dir / NONSTANDARD_UNET_FILENAME
    )
    base_variant = model_info.get("base_variant")

    unet = UNet2DConditionModel.from_config(model_info["unet_config"]).to(
        dtype=weight_dtype
    )
    unet_missing, unet_unexpected = load_model(
        unet,
        str(unet_weights_path),
        strict=True,
    )
    assert not unet_missing and not unet_unexpected

    brushnet = BrushNetModel.from_unet(
        unet, load_weights_from_unet=False
    ).to(dtype=weight_dtype)
    brushnet_missing, brushnet_unexpected = load_model(
        brushnet,
        str(weights_dir / BRUSHNET_SAFETENSORS_FILENAME),
        strict=True,
    )
    assert not brushnet_missing and not brushnet_unexpected

    base_pretrained_kwargs = {
        "torch_dtype": weight_dtype,
        "use_safetensors": True,
        "local_files_only": True,
    }
    if base_variant is not None:
        base_pretrained_kwargs["variant"] = base_variant

    vae = AutoencoderKL.from_pretrained(
        str(base_dir),
        subfolder="vae",
        **base_pretrained_kwargs,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(base_dir),
        subfolder="text_encoder",
        **base_pretrained_kwargs,
    )
    # Tokenizer, scheduler, and text_encoder_brushnet keep realisticVisionV60B1_v51VAE /
    # PowerPaint_Brushnet because AnyLoRA and realistic share the exact same SD1.5 CLIP
    # tokenizer/scheduler configuration, while text_encoder_brushnet holds PowerPaint's
    # learned task token embeddings.
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        str(realistic_base_dir),
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    tokenizer = TokenizerWrapper(
        from_pretrained=str(realistic_base_dir),
        subfolder="tokenizer",
        revision=None,
        local_files_only=True,
    )
    add_tokens(
        tokenizer=tokenizer,
        text_encoder=text_encoder_brushnet,
        placeholder_tokens=list(PLACEHOLDER_TOKENS),
        initialize_tokens=list(INITIALIZE_TOKENS),
        num_vectors_per_token=NUM_VECTORS_PER_TOKEN,
    )

    task_state = gate.torch.load(
        str(weights_dir / TASK_TEXT_ENCODER_BIN_FILENAME),
        map_location="cpu",
        weights_only=True,
    )
    load_task_text_encoder_state(text_encoder_brushnet, task_state, gate.torch)
    text_encoder_brushnet = text_encoder_brushnet.to(dtype=weight_dtype)
    for ext_entry in (
        text_encoder_brushnet.text_model.embeddings.token_embedding.external_embeddings
    ):
        name = ext_entry["name"]
        ext_entry["embedding"] = (
            text_encoder_brushnet.text_model.embeddings.token_embedding.trainable_embeddings[
                name
            ]
        )

    scheduler = UniPCMultistepScheduler.from_config(model_info["scheduler_config"])
    pipe = StableDiffusionPowerPaintBrushNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_brushnet=text_encoder_brushnet,
        tokenizer=tokenizer,
        unet=unet,
        brushnet=brushnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    assert pipe.safety_checker is None
    if lora is not None:
        load_brushnet_lora(
            pipe.brushnet,
            gate.LAB,
            lora,
            dtype=weight_dtype,
            lora_info=(
                model_info.get("lora_info")
                if isinstance(model_info, dict)
                else None
            ),
        )
    if offload:
        pipe.model_cpu_offload_seq = MODEL_CPU_OFFLOAD_SEQ
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    return pipe


def release_pipeline(pipe, gate):
    """Free accelerate offload hooks, pipeline references, Python GC, and CUDA cache."""
    if pipe is not None:
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
    del pipe
    gc.collect()
    if gate.torch.cuda.is_available():
        gate.torch.cuda.empty_cache()


def _validate_lora_run_arg(value):
    if not isinstance(value, str) or not RUN_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"Invalid LoRA run name: {value!r}"
        )
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_name",
        help="Fresh output directory name under .local-artifacts/outpainting",
    )
    parser.add_argument(
        "--dataset",
        choices=("evaluation", "extended-evaluation-v2"),
        default="evaluation",
        help="Frozen evaluation dataset to run (default: evaluation)",
    )
    parser.add_argument(
        "--base",
        choices=("realistic", "anylora"),
        default="realistic",
        help="SD1.5 base model to pair with PowerPaint BrushNet (default: realistic)",
    )
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES_PP.keys()),
        default="lab",
        help="Outpainting recipe to run (default: lab)",
    )
    parser.add_argument(
        "--lora",
        type=_validate_lora_run_arg,
        default=None,
        help="Optional trained BrushNet LoRA run name under .local-artifacts/outpainting/lora-runs",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override seeds (default: 42 and 123 with --full, else 42)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all evaluation outputs across dataset groups, 4 directions, and 2 seeds",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    recipe_spec = RECIPES_PP[args.recipe]

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(LAB / "cache")

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    model_info = verify_powerpaint_preflight(LAB, base=args.base)
    lora_info = (
        verify_lora_preflight(LAB, args.lora)
        if args.lora is not None
        else {"lora_run": None, "lora_sha256": None}
    )
    model_info["lora_info"] = lora_info if args.lora is not None else None
    all_sources, evaluation_manifest_sha256 = evaluation_sources(
        LAB, dataset=args.dataset
    )
    _ensure_vendor_runtime_paths(LAB)

    import placement_poc as placement
    import quality_gate as gate

    if not gate.torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run PowerPaint v2 GPU evaluation")
    gate.torch.set_grad_enabled(False)

    with gate.single_instance():
        folder = new_output(gate.LAB, args.run_name)
        gpu_name = gate.torch.cuda.get_device_name(0)

        group_order = tuple(all_sources)
        selected_groups = group_order if args.full else (group_order[-1],)
        selected_directions = (
            ("left", "top", "right", "bottom") if args.full else ("bottom",)
        )
        selected_seeds = tuple(args.seeds) if args.seeds else ((42, 123) if args.full else (42,))
        total_expected = (
            len(selected_groups) * len(selected_directions) * len(selected_seeds)
        )

        cases = []
        for group in selected_groups:
            source_path = all_sources[group]
            full_sha256 = sha256_file(source_path)
            with Image.open(source_path) as raw_image:
                full = ImageOps.exif_transpose(raw_image).convert("RGB")
            for direction in selected_directions:
                crop_box, margins = direction_crop_and_margins(full.size, direction)
                source = full.crop(crop_box)
                assert source.mode == "RGB"
                source_bytes = source.tobytes()
                source_rgb_sha256 = hashlib.sha256(source_bytes).hexdigest()

                canvas, mask, box = placement.prepare(source, margins)
                pipeline_mask = expand_mask(mask, recipe_spec["mask_overlap_px"])
                masked_canvas = prepare_brushnet_input(canvas, pipeline_mask)
                output_box = tuple(round(v * 2.5) for v in box)
                seams = seam_crop_boxes(output_box)

                case_dir = folder / f"{group}-{direction}"
                case_dir.mkdir(parents=True, exist_ok=False)
                source.save(case_dir / "source.png")
                canvas.save(case_dir / "canvas.png")
                mask.save(case_dir / "mask.png")
                if args.recipe == "official":
                    pipeline_mask.save(case_dir / "mask-official.png")
                full.save(case_dir / "full-reference-evaluation-only.png")

                cases.append(
                    {
                        "group": group,
                        "direction": direction,
                        "name": f"{group}-{direction}",
                        "dir": case_dir,
                        "path": source_path,
                        "full_sha256": full_sha256,
                        "full_size": full.size,
                        "source": source,
                        "source_bytes": source_bytes,
                        "source_rgb_sha256": source_rgb_sha256,
                        "crop_box": crop_box,
                        "margins": margins,
                        "canvas": canvas,
                        "canvas_bytes": canvas.tobytes(),
                        "masked_canvas": masked_canvas,
                        "masked_canvas_bytes": masked_canvas.tobytes(),
                        "mask": mask,
                        "mask_bytes": mask.tobytes(),
                        "pipeline_mask": pipeline_mask,
                        "pipeline_mask_bytes": pipeline_mask.tobytes(),
                        "box": box,
                        "output_box": output_box,
                        "seam_boxes": seams,
                        "records": [],
                    }
                )

        records = []
        pipe = None
        try:
            pipe = load_powerpaint_pipeline(
                gate, model_info=model_info, base=args.base, lora=args.lora
            )
            assert pipe.safety_checker is None

            for case in cases:
                for seed in selected_seeds:
                    assert case["source"].tobytes() == case["source_bytes"]
                    assert case["canvas"].tobytes() == case["canvas_bytes"]
                    assert case["mask"].tobytes() == case["mask_bytes"]
                    assert (
                        case["pipeline_mask"].tobytes()
                        == case["pipeline_mask_bytes"]
                    )

                    masked_input = prepare_brushnet_input(
                        case["canvas"], case["pipeline_mask"]
                    )
                    assert masked_input.tobytes() == case["masked_canvas_bytes"]
                    assert case["canvas"].tobytes() == case["canvas_bytes"]
                    assert case["mask"].tobytes() == case["mask_bytes"]
                    assert (
                        case["pipeline_mask"].tobytes()
                        == case["pipeline_mask_bytes"]
                    )
                    if recipe_spec["mask_overlap_px"] == 0:
                        assert (
                            ImageChops.difference(
                                masked_input.crop(case["box"]),
                                case["canvas"].crop(case["box"]),
                            ).getbbox()
                            is None
                        )
                    rgb_mask = case["pipeline_mask"].convert("RGB")

                    gate.torch.cuda.reset_peak_memory_stats()
                    gate.torch.cuda.synchronize()
                    started = perf_counter()
                    result = pipe(
                        promptA=PROMPT_A,
                        promptB=PROMPT_B,
                        promptU=recipe_spec["promptU"],
                        tradoff=TRADOFF,
                        tradoff_nag=TRADOFF_NAG,
                        image=masked_input,
                        mask=rgb_mask,
                        num_inference_steps=recipe_spec["num_inference_steps"],
                        generator=gate.torch.Generator(device="cpu").manual_seed(
                            seed
                        ),
                        brushnet_conditioning_scale=BRUSHNET_CONDITIONING_SCALE,
                        negative_promptA=NEGATIVE_PROMPT_A,
                        negative_promptB=NEGATIVE_PROMPT_B,
                        negative_promptU=recipe_spec["negative_promptU"],
                        guidance_scale=recipe_spec["guidance_scale"],
                        width=INFERENCE_WIDTH,
                        height=INFERENCE_HEIGHT,
                    )
                    gate.torch.cuda.synchronize()
                    seconds = perf_counter() - started

                    assert pipe.safety_checker is None
                    assert result.nsfw_content_detected is None
                    generated = result.images[0]
                    assert generated.size == (INFERENCE_WIDTH, INFERENCE_HEIGHT)

                    composited = Image.composite(
                        generated, case["canvas"], case["pipeline_mask"]
                    )
                    if recipe_spec["mask_overlap_px"] == 0:
                        assert (
                            ImageChops.difference(
                                composited.crop(case["box"]),
                                case["canvas"].crop(case["box"]),
                            ).getbbox()
                            is None
                        )
                    output = placement.restore(
                        case["source"], composited, case["box"]
                    )

                    stem = f"{RECIPE}-{seed}"
                    generated.save(case["dir"] / f"{stem}-generated.png")
                    output_path = case["dir"] / f"{stem}.png"
                    output.save(output_path)

                    with Image.open(output_path) as decoded:
                        decoded.load()
                        assert decoded.format == "PNG"
                        assert decoded.size == (1920, 1080)

                    obox = case["output_box"]
                    fitted = case["source"].resize(
                        (obox[2] - obox[0], obox[3] - obox[1]),
                        Image.Resampling.LANCZOS,
                    )
                    assert (
                        ImageChops.difference(
                            output.crop(obox), fitted
                        ).getbbox()
                        is None
                    )
                    assert sha256_file(case["path"]) == case["full_sha256"]
                    assert (
                        hashlib.sha256(case["source"].tobytes()).hexdigest()
                        == case["source_rgb_sha256"]
                    )

                    for edge, sbox in case["seam_boxes"].items():
                        output.crop(tuple(sbox)).save(
                            case["dir"] / f"{stem}-seam100-{edge}.png"
                        )

                    peak_alloc_mib = (
                        gate.torch.cuda.max_memory_allocated() / 1048576
                    )
                    peak_reserved_mib = (
                        gate.torch.cuda.max_memory_reserved() / 1048576
                    )

                    record = {
                        "dataset": args.dataset,
                        "evaluation_manifest_sha256": evaluation_manifest_sha256,
                        "group": case["group"],
                        "direction": case["direction"],
                        "case": case["name"],
                        "seed": seed,
                        "recipe": RECIPE,
                        "recipe_name": args.recipe,
                        "lora_run": lora_info["lora_run"],
                        "lora_sha256": lora_info["lora_sha256"],
                        "lora_weights_sha256": lora_info["lora_sha256"],
                        "brushnet_lora_sha256": lora_info["lora_sha256"],
                        "base_name": model_info["base_name"],
                        "base_repository": model_info["base_repository"],
                        "base_revision": model_info["base_revision"],
                        "base_weights_sha256": model_info["base_weights_sha256"],
                        "model_repository": model_info["repository"],
                        "model_revision": model_info["revision"],
                        "code_revision": model_info["code_revision"],
                        "manifest_sha256": model_info["manifest_sha256"],
                        "weights_manifest_sha256": model_info["manifest_sha256"],
                        "weights_hashes": model_info["hashes"],
                        "scheduler_class": "UniPCMultistepScheduler",
                        "scheduler_config_path": model_info["scheduler_config_path"],
                        "scheduler_config_sha256": model_info[
                            "scheduler_config_sha256"
                        ],
                        "steps": recipe_spec["num_inference_steps"],
                        "num_inference_steps": recipe_spec["num_inference_steps"],
                        "cfg": recipe_spec["guidance_scale"],
                        "guidance_scale": recipe_spec["guidance_scale"],
                        "mask_overlap_px": recipe_spec["mask_overlap_px"],
                        "brushnet_conditioning_scale": BRUSHNET_CONDITIONING_SCALE,
                        "tradoff": TRADOFF,
                        "tradoff_nag": TRADOFF_NAG,
                        "prompt": recipe_spec["promptU"],
                        "negative_prompt": recipe_spec["negative_promptU"],
                        "promptU": recipe_spec["promptU"],
                        "negative_promptU": recipe_spec["negative_promptU"],
                        "promptA": PROMPT_A,
                        "promptB": PROMPT_B,
                        "negative_promptA": NEGATIVE_PROMPT_A,
                        "negative_promptB": NEGATIVE_PROMPT_B,
                        "task_tokens": list(PLACEHOLDER_TOKENS),
                        "num_vectors_per_token": NUM_VECTORS_PER_TOKEN,
                        "source_full_sha256": case["full_sha256"],
                        "source_sha256": case["full_sha256"],
                        "cropped_rgb_sha256": case["source_rgb_sha256"],
                        "source_full_size": list(case["full_size"]),
                        "cropped_source_size": list(case["source"].size),
                        "crop_box": list(case["crop_box"]),
                        "margins": list(case["margins"]),
                        "box": list(case["box"]),
                        "output_box": list(case["output_box"]),
                        "seam_crop_boxes": case["seam_boxes"],
                        "evaluation_only_full_reference": "full-reference-evaluation-only.png",
                        "torch_version": gate.torch.__version__,
                        "diffusers_version": gate.diffusers.__version__,
                        "cuda_version": gate.torch.version.cuda,
                        "gpu_name": gpu_name,
                        "seconds": seconds,
                        "elapsed_seconds": seconds,
                        "torch_allocated_peak_mib": peak_alloc_mib,
                        "torch_reserved_peak_mib": peak_reserved_mib,
                        "foreground_exact": True,
                        "safety_checker_none": True,
                        "nsfw_content_detected": result.nsfw_content_detected,
                        "result_nsfw_content_detected": result.nsfw_content_detected,
                    }
                    case["records"].append(record)
                    records.append(record)
                    (case["dir"] / "records.json").write_text(
                        json.dumps(case["records"], indent=2), encoding="utf-8"
                    )
                    (folder / "records.json").write_text(
                        json.dumps(records, indent=2), encoding="utf-8"
                    )
                    print(
                        f"{len(records)}/{total_expected} [{RECIPE}] {case['name']} {stem} ({seconds:.2f}s)",
                        flush=True,
                    )
        finally:
            release_pipeline(pipe, gate)
            pipe = None
            gc.collect()
            if gate.torch.cuda.is_available():
                gate.torch.cuda.empty_cache()

        print(
            f"Completed {len(records)} records for {RECIPE} in {folder}",
            flush=True,
        )


if __name__ == "__main__":
    main()
