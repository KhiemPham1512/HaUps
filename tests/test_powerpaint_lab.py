import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPAINT_DIR = REPO_ROOT / ".local-artifacts" / "outpainting"
LAB_CODE = REPO_ROOT / "lab" / "outpainting"

placement_spec = importlib.util.spec_from_file_location(
    "placement_poc", LAB_CODE / "placement_poc.py"
)
placement = importlib.util.module_from_spec(placement_spec)
placement_spec.loader.exec_module(placement)

powerpaint_spec = importlib.util.spec_from_file_location(
    "powerpaint_gpu", LAB_CODE / "powerpaint_gpu.py"
)
powerpaint = importlib.util.module_from_spec(powerpaint_spec)
powerpaint_spec.loader.exec_module(powerpaint)


def test_import_stays_lightweight_without_torch_or_vendor_modules():
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules
    assert "powerpaint" not in sys.modules
    vendor_str = str(OUTPAINT_DIR / "powerpaint-v2" / "vendor")
    deps_str = str(OUTPAINT_DIR / "powerpaint-v2" / "dependencies")
    assert vendor_str not in sys.path
    assert deps_str not in sys.path


def test_parse_args_accepts_only_run_name_and_full_flag():
    smoke = powerpaint.parse_args(["pp-smoke-01"])
    assert smoke.run_name == "pp-smoke-01"
    assert smoke.dataset == "evaluation"
    assert smoke.base == "realistic"
    assert smoke.recipe == "lab"
    assert smoke.lora is None
    assert smoke.full is False

    lora_args = powerpaint.parse_args(
        ["pp-lora-01", "--lora", "brushnet-lora-01"]
    )
    assert lora_args.run_name == "pp-lora-01"
    assert lora_args.lora == "brushnet-lora-01"

    full = powerpaint.parse_args(
        ["pp-full-01", "--dataset", "extended-evaluation-v2", "--full"]
    )
    assert full.run_name == "pp-full-01"
    assert full.dataset == "extended-evaluation-v2"
    assert full.base == "realistic"
    assert full.recipe == "lab"
    assert full.full is True

    eval_explicit = powerpaint.parse_args(
        ["pp-eval-01", "--dataset", "evaluation", "--base", "realistic"]
    )
    assert eval_explicit.dataset == "evaluation"
    assert eval_explicit.base == "realistic"
    assert eval_explicit.recipe == "lab"

    anylora_args = powerpaint.parse_args(
        ["pp-anylora-01", "--base", "anylora"]
    )
    assert anylora_args.run_name == "pp-anylora-01"
    assert anylora_args.base == "anylora"
    assert anylora_args.dataset == "evaluation"
    assert anylora_args.recipe == "lab"
    assert anylora_args.full is False

    official_args = powerpaint.parse_args(
        ["pp-official-01", "--recipe", "official"]
    )
    assert official_args.run_name == "pp-official-01"
    assert official_args.recipe == "official"
    assert official_args.base == "realistic"
    assert official_args.dataset == "evaluation"
    assert official_args.full is False

    lab_explicit = powerpaint.parse_args(
        ["pp-lab-01", "--recipe", "lab"]
    )
    assert lab_explicit.recipe == "lab"

    for disallowed in (
        ["pp-run", "--model", "sd15"],
        ["pp-run", "--base", "sdxl"],
        ["pp-run", "--base", "../anylora"],
        ["pp-run", "--base", "/tmp/anylora"],
        ["pp-run", "--recipe", "custom"],
        ["pp-run", "--recipe", "v1"],
        ["pp-run", "--recipe", "../official"],
        ["pp-run", "--dataset", "development"],
        ["pp-run", "--dataset", "extended-evaluation"],
        ["pp-run", "--dataset", "../evaluation"],
        ["pp-run", "--lora", "../escape"],
        ["pp-run", "--lora", "/tmp/lora"],
        ["pp-run", "--lora", "sub/lora"],
        ["pp-run", "--lora", "-bad"],
        ["pp-run", "--lora", ""],
        ["pp-run", "--source", "custom.png"],
        ["pp-run", "extra-arg"],
    ):
        with pytest.raises(SystemExit):
            powerpaint.parse_args(disallowed)


def test_recipes_pp_specs_and_official_prompt_suffix():
    assert set(powerpaint.RECIPES_PP.keys()) == {"lab", "official"}
    lab_recipe = powerpaint.RECIPES_PP["lab"]
    official_recipe = powerpaint.RECIPES_PP["official"]

    assert lab_recipe["recipe_name"] == "lab"
    assert lab_recipe["guidance_scale"] == 7.0
    assert lab_recipe["num_inference_steps"] == 30
    assert lab_recipe["mask_overlap_px"] == 0
    assert not lab_recipe["promptU"].endswith("empty scene")

    assert official_recipe["recipe_name"] == "official"
    assert official_recipe["guidance_scale"] == 10.0
    assert official_recipe["num_inference_steps"] == 45
    assert official_recipe["mask_overlap_px"] == 8
    assert official_recipe["promptU"].endswith("empty scene")
    assert official_recipe["promptU"] == f"{lab_recipe['promptU']}, empty scene"
    assert official_recipe["negative_promptU"] == lab_recipe["negative_promptU"]


def test_expand_mask_shrinks_zero_box_by_px_on_adjacent_255_edges_and_preserves_zero_px():
    width, height = 120, 90
    box = (20, 15, 100, 75)
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, box)
    mask_bytes_before = mask.tobytes()

    same_mask = powerpaint.expand_mask(mask, 0)
    assert same_mask.mode == "L"
    assert same_mask.size == mask.size
    assert same_mask.tobytes() == mask_bytes_before
    assert mask.tobytes() == mask_bytes_before

    px = 8
    expanded = powerpaint.expand_mask(mask, px)
    assert expanded.mode == "L"
    assert expanded.size == (width, height)
    assert mask.tobytes() == mask_bytes_before

    expected_zero_box = (box[0] + px, box[1] + px, box[2] - px, box[3] - px)
    expected = Image.new("L", (width, height), 255)
    expected.paste(0, expected_zero_box)
    assert ImageChops.difference(expanded, expected).getbbox() is None

    inverted = ImageChops.invert(expanded)
    assert inverted.getbbox() == expected_zero_box

    for direction in ("left", "top", "right", "bottom"):
        _, margins = powerpaint.direction_crop_and_margins((120, 90), direction)
        source = Image.new("RGB", (96, 72), (40, 80, 120))
        _, dir_mask, dir_box = placement.prepare(source, margins)
        dir_expanded = powerpaint.expand_mask(dir_mask, px)
        assert dir_expanded.mode == "L"
        assert dir_expanded.size == dir_mask.size

        x0, y0, x1, y1 = dir_box
        exp_x0 = x0 + px if x0 > 0 else x0
        exp_y0 = y0 + px if y0 > 0 else y0
        exp_x1 = x1 - px if x1 < dir_mask.width else x1
        exp_y1 = y1 - px if y1 < dir_mask.height else y1
        assert ImageChops.invert(dir_expanded).getbbox() == (
            exp_x0,
            exp_y0,
            exp_x1,
            exp_y1,
        )

    with pytest.raises(ValueError, match="L-mode"):
        powerpaint.expand_mask(Image.new("RGB", (32, 32)), 8)
    with pytest.raises(ValueError, match="integer"):
        powerpaint.expand_mask(mask, -1)


@pytest.mark.parametrize("direction", ["left", "top", "right", "bottom"])
def test_prepare_brushnet_input_preserves_known_pixels_and_zeros_masked_region(
    direction,
):
    source = Image.new("RGB", (96, 72), (35, 120, 210))
    source.putpixel((0, 0), (250, 20, 30))
    source.putpixel((95, 71), (20, 240, 60))
    source_before = source.tobytes()

    _, margins = powerpaint.direction_crop_and_margins((120, 90), direction)
    canvas, mask, box = placement.prepare(source, margins)
    canvas_before = canvas.tobytes()
    mask_before = mask.tobytes()

    masked_canvas = powerpaint.prepare_brushnet_input(canvas, mask)

    assert masked_canvas.mode == "RGB"
    assert masked_canvas.size == (768, 432)
    assert source.tobytes() == source_before
    assert canvas.tobytes() == canvas_before
    assert mask.tobytes() == mask_before

    assert (
        ImageChops.difference(
            masked_canvas.crop(box), canvas.crop(box)
        ).getbbox()
        is None
    )

    x0, y0, x1, y1 = box
    outside_probes = []
    if x0 > 0:
        outside_probes.extend([(0, y0), (x0 - 1, (y0 + y1) // 2)])
    if y0 > 0:
        outside_probes.extend([(x0, 0), ((x0 + x1) // 2, y0 - 1)])
    if x1 < 768:
        outside_probes.extend([(x1, y0), (767, (y0 + y1) // 2)])
    if y1 < 432:
        outside_probes.extend([(x0, y1), ((x0 + x1) // 2, 431)])
    assert outside_probes
    for px, py in outside_probes:
        assert mask.getpixel((px, py)) == 255
        assert canvas.getpixel((px, py)) == (255, 255, 255)
        assert masked_canvas.getpixel((px, py)) == (0, 0, 0)

    expected = Image.new("RGB", canvas.size, (0, 0, 0))
    expected.paste(canvas.crop(box), box)
    assert ImageChops.difference(masked_canvas, expected).getbbox() is None
    assert (
        ImageChops.difference(
            masked_canvas, Image.new("RGB", canvas.size, (0, 0, 0))
        ).getbbox()
        == box
    )

    masked_canvas.putpixel((x0, y0), (0, 0, 0))
    assert canvas.tobytes() == canvas_before


def test_prepare_brushnet_input_rejects_invalid_modes_or_size_mismatch():
    valid_canvas = Image.new("RGB", (64, 48), "white")
    valid_mask = Image.new("L", (64, 48), 255)

    with pytest.raises(ValueError, match="RGB"):
        powerpaint.prepare_brushnet_input(Image.new("L", (64, 48)), valid_mask)
    with pytest.raises(ValueError, match="L-mode"):
        powerpaint.prepare_brushnet_input(
            valid_canvas, Image.new("RGB", (64, 48))
        )
    with pytest.raises(ValueError, match="dimensions"):
        powerpaint.prepare_brushnet_input(
            valid_canvas, Image.new("L", (32, 48), 255)
        )


def _write_fake_weights_tree(lab_root: Path):
    weights_dir = lab_root / "powerpaint-v2" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = [
        "realisticVisionV60B1_v51VAE/model_index.json",
        "realisticVisionV60B1_v51VAE/scheduler/scheduler_config.json",
        "realisticVisionV60B1_v51VAE/tokenizer/merges.txt",
        "realisticVisionV60B1_v51VAE/tokenizer/special_tokens_map.json",
        "realisticVisionV60B1_v51VAE/tokenizer/tokenizer_config.json",
        "realisticVisionV60B1_v51VAE/tokenizer/vocab.json",
        "realisticVisionV60B1_v51VAE/text_encoder/config.json",
        "realisticVisionV60B1_v51VAE/vae/config.json",
        "realisticVisionV60B1_v51VAE/unet/config.json",
        "PowerPaint_Brushnet/config.json",
        *powerpaint.MANDATORY_WEIGHT_FILES.keys(),
    ]
    assert len(rel_paths) == powerpaint.EXPECTED_WEIGHT_FILE_COUNT
    entries = []
    for rel in rel_paths:
        target = weights_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"stub": rel}).encode("utf-8")
        target.write_bytes(payload)
        entries.append(
            {
                "file": rel,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "complete": True,
        "repo": powerpaint.POWERPAINT_REPOSITORY,
        "revision": powerpaint.POWERPAINT_MODEL_REVISION,
        "files": entries,
    }
    manifest_path = weights_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return weights_dir, manifest_path, manifest


def test_verify_powerpaint_preflight_rejects_incomplete_or_bad_metadata(tmp_path):
    _, manifest_path, base_manifest = _write_fake_weights_tree(tmp_path)

    incomplete = dict(base_manifest, complete=False)
    manifest_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    bad_repo = dict(base_manifest, repo="other/repo")
    manifest_path.write_text(json.dumps(bad_repo), encoding="utf-8")
    with pytest.raises(ValueError, match="repository"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    bad_rev = dict(base_manifest, revision="0" * 40)
    manifest_path.write_text(json.dumps(bad_rev), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    short_files = dict(base_manifest, files=base_manifest["files"][:-1])
    manifest_path.write_text(json.dumps(short_files), encoding="utf-8")
    with pytest.raises(ValueError, match="15 files"):
        powerpaint.verify_powerpaint_preflight(tmp_path)


def test_verify_powerpaint_preflight_rejects_unsafe_paths_and_hash_or_size_mismatch(
    tmp_path,
):
    weights_dir, manifest_path, base_manifest = _write_fake_weights_tree(tmp_path)

    for bad_rel in ("../outside.bin", "/abs/file.bin", "sub\\backslash.json"):
        escaped = dict(base_manifest)
        escaped_files = [dict(e) for e in base_manifest["files"]]
        escaped_files[0]["file"] = bad_rel
        escaped["files"] = escaped_files
        manifest_path.write_text(json.dumps(escaped), encoding="utf-8")
        with pytest.raises(ValueError):
            powerpaint.verify_powerpaint_preflight(tmp_path)

    bad_sha = dict(base_manifest)
    bad_sha_files = [dict(e) for e in base_manifest["files"]]
    bad_sha_files[0]["sha256"] = "0" * 64
    bad_sha["files"] = bad_sha_files
    manifest_path.write_text(json.dumps(bad_sha), encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    bad_size = dict(base_manifest)
    bad_size_files = [dict(e) for e in base_manifest["files"]]
    bad_size_files[0]["bytes"] = 99999
    bad_size["files"] = bad_size_files
    manifest_path.write_text(json.dumps(bad_size), encoding="utf-8")
    with pytest.raises(AssertionError, match="Size mismatch"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="Pinned size mismatch"):
        powerpaint.verify_powerpaint_preflight(tmp_path)

    symlink_target = tmp_path / "external.json"
    symlink_target.write_bytes(b"{}")
    first_file = weights_dir / base_manifest["files"][0]["file"]
    first_file.unlink()
    try:
        first_file.symlink_to(symlink_target)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="Symlink"):
            powerpaint.verify_powerpaint_preflight(tmp_path)


def test_load_task_text_encoder_state_rejects_missing_or_unchanged_task_tensors():
    class FakeTensor:
        def __init__(self, value, shape=(10, 768), finite=True):
            self.value = value
            self.shape = shape
            self._finite = finite
            self.device = "cpu"
            self.dtype = "fp32"

        def numel(self):
            total = 1
            for dim in self.shape:
                total *= dim
            return total

        def all(self):
            return self._finite

        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return FakeTensor(self.value, self.shape, self._finite)

        def float(self):
            return self

        def to(self, device=None, dtype=None):
            return FakeTensor(self.value, self.shape, self._finite)

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def isfinite(tensor):
            return tensor

        @staticmethod
        def allclose(a, b):
            return a.value == b.value and a.shape == b.shape

    class FakeEmbWrapper:
        def __init__(self):
            self.trainable_embeddings = {
                t: FakeTensor(0.0) for t in powerpaint.PLACEHOLDER_TOKENS
            }
            self.external_embeddings = [
                {"name": t, "embedding": self.trainable_embeddings[t]}
                for t in powerpaint.PLACEHOLDER_TOKENS
            ]

    class FakeTextEncoder:
        def __init__(self):
            self.emb = FakeEmbWrapper()
            self.text_model = type(
                "TM",
                (),
                {"embeddings": type("E", (), {"token_embedding": self.emb})()},
            )()
            self._state = {
                "text_model.embeddings.token_embedding.wrapped.weight": FakeTensor(
                    1.0, (100, 768)
                ),
                **{
                    f"text_model.embeddings.token_embedding.trainable_embeddings.{t}": self.emb.trainable_embeddings[
                        t
                    ]
                    for t in powerpaint.PLACEHOLDER_TOKENS
                },
            }

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state_dict, strict=True):
            for token in powerpaint.PLACEHOLDER_TOKENS:
                key = f"text_model.embeddings.token_embedding.trainable_embeddings.{token}"
                if key in state_dict:
                    self.emb.trainable_embeddings[token].value = state_dict[
                        key
                    ].value
            return type(
                "Incompat", (), {"missing_keys": [], "unexpected_keys": []}
            )()

    encoder = FakeTextEncoder()
    with pytest.raises(ValueError, match="Missing required task token tensor"):
        powerpaint.load_task_text_encoder_state(
            encoder,
            {"only_other": FakeTensor(1.0)},
            FakeTorch,
        )

    unchanged_state = encoder.state_dict()
    with pytest.raises(ValueError, match="placeholder initialization"):
        powerpaint.load_task_text_encoder_state(
            encoder, unchanged_state, FakeTorch
        )

    valid_state = {
        "text_model.embeddings.token_embedding.wrapped.weight": FakeTensor(
            1.0, (100, 768)
        ),
        **{
            f"text_model.embeddings.token_embedding.trainable_embeddings.{t}": FakeTensor(
                float(i + 2)
            )
            for i, t in enumerate(powerpaint.PLACEHOLDER_TOKENS)
        },
    }
    powerpaint.load_task_text_encoder_state(encoder, valid_state, FakeTorch)
    for i, token in enumerate(powerpaint.PLACEHOLDER_TOKENS):
        assert encoder.emb.trainable_embeddings[token].value == float(i + 2)


def test_load_task_text_encoder_state_filters_verified_legacy_position_ids_and_rejects_invalid_or_unexpected():
    class FakeTensor:
        def __init__(self, value, shape=(10, 768), finite=True, dtype=None):
            self.value = value
            self.shape = shape
            self._finite = finite
            self.device = "cpu"
            if dtype is None:
                if isinstance(value, (list, tuple)) and all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in value
                ):
                    dtype = "int64"
                else:
                    dtype = "fp32"
            self.dtype = dtype

        def numel(self):
            total = 1
            for dim in self.shape:
                total *= dim
            return total

        def all(self):
            return self._finite

        def is_floating_point(self):
            return "float" in self.dtype or "fp" in self.dtype

        def is_complex(self):
            return "complex" in self.dtype

        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return FakeTensor(
                self.value, self.shape, self._finite, dtype=self.dtype
            )

        def float(self):
            return self

        def reshape(self, *dims):
            return self

        def tolist(self):
            if isinstance(self.value, tuple):
                return list(self.value)
            if isinstance(self.value, list):
                return list(self.value)
            return [self.value]

        def to(self, device=None, dtype=None):
            return FakeTensor(
                self.value,
                self.shape,
                self._finite,
                dtype=dtype or self.dtype,
            )

    class FakeTorch:
        Tensor = FakeTensor

        @staticmethod
        def isfinite(tensor):
            return tensor

        @staticmethod
        def allclose(a, b):
            return a.value == b.value and a.shape == b.shape

    class FakeEmbWrapper:
        def __init__(self):
            self.trainable_embeddings = {
                t: FakeTensor(0.0) for t in powerpaint.PLACEHOLDER_TOKENS
            }
            self.external_embeddings = [
                {"name": t, "embedding": self.trainable_embeddings[t]}
                for t in powerpaint.PLACEHOLDER_TOKENS
            ]

    class FakeTextEncoder:
        def __init__(self, max_positions=77, expect_position_ids_in_state=False):
            self.emb = FakeEmbWrapper()
            self.runtime_position_ids = FakeTensor(
                list(range(max_positions)),
                (1, max_positions),
                dtype="int64",
            )
            self.text_model = type(
                "TM",
                (),
                {
                    "embeddings": type(
                        "E",
                        (),
                        {
                            "token_embedding": self.emb,
                            "position_ids": self.runtime_position_ids,
                        },
                    )()
                },
            )()
            self._state = {
                "text_model.embeddings.token_embedding.wrapped.weight": FakeTensor(
                    1.0, (100, 768)
                ),
                **{
                    f"text_model.embeddings.token_embedding.trainable_embeddings.{t}": self.emb.trainable_embeddings[
                        t
                    ]
                    for t in powerpaint.PLACEHOLDER_TOKENS
                },
            }
            if expect_position_ids_in_state:
                self._state[powerpaint.LEGACY_POSITION_IDS_KEY] = (
                    self.runtime_position_ids
                )
            self.load_calls = []

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state_dict, strict=True):
            self.load_calls.append((dict(state_dict), strict))
            if not strict:
                raise AssertionError(
                    "Blanket strict=False is not allowed for task text encoder loading"
                )
            if set(state_dict.keys()) != set(self._state.keys()):
                raise RuntimeError(
                    f"Strict state_dict mismatch: {set(state_dict.keys()) ^ set(self._state.keys())}"
                )
            for token in powerpaint.PLACEHOLDER_TOKENS:
                key = f"text_model.embeddings.token_embedding.trainable_embeddings.{token}"
                self.emb.trainable_embeddings[token].value = state_dict[
                    key
                ].value
            return type(
                "Incompat", (), {"missing_keys": [], "unexpected_keys": []}
            )()

    def _base_valid_state():
        return {
            "text_model.embeddings.token_embedding.wrapped.weight": FakeTensor(
                1.0, (100, 768)
            ),
            **{
                f"text_model.embeddings.token_embedding.trainable_embeddings.{t}": FakeTensor(
                    float(i + 2)
                )
                for i, t in enumerate(powerpaint.PLACEHOLDER_TOKENS)
            },
        }

    # 1. Runtime omits position_ids in state_dict(), checkpoint has valid legacy position_ids buffer
    encoder = FakeTextEncoder(max_positions=77, expect_position_ids_in_state=False)
    legacy_state = {
        **_base_valid_state(),
        powerpaint.LEGACY_POSITION_IDS_KEY: FakeTensor(
            list(range(77)), (1, 77), dtype="int64"
        ),
    }
    snapshot_keys = set(legacy_state.keys())
    powerpaint.load_task_text_encoder_state(encoder, legacy_state, FakeTorch)
    assert set(legacy_state.keys()) == snapshot_keys
    assert powerpaint.LEGACY_POSITION_IDS_KEY in legacy_state
    assert len(encoder.load_calls) == 1
    loaded_dict, strict_flag = encoder.load_calls[0]
    assert strict_flag is True
    assert powerpaint.LEGACY_POSITION_IDS_KEY not in loaded_dict
    for i, token in enumerate(powerpaint.PLACEHOLDER_TOKENS):
        assert encoder.emb.trainable_embeddings[token].value == float(i + 2)

    # 2. Runtime expects position_ids in state_dict(): preserve key and strict loading
    strict_encoder = FakeTextEncoder(
        max_positions=77, expect_position_ids_in_state=True
    )
    powerpaint.load_task_text_encoder_state(
        strict_encoder, legacy_state, FakeTorch
    )
    assert len(strict_encoder.load_calls) == 1
    strict_loaded_dict, strict_flag = strict_encoder.load_calls[0]
    assert strict_flag is True
    assert powerpaint.LEGACY_POSITION_IDS_KEY in strict_loaded_dict

    strict_missing_pos_encoder = FakeTextEncoder(
        max_positions=77, expect_position_ids_in_state=True
    )
    missing_pos_state = _base_valid_state()
    missing_pos_snapshot_keys = set(missing_pos_state.keys())
    powerpaint.load_task_text_encoder_state(
        strict_missing_pos_encoder, missing_pos_state, FakeTorch
    )
    assert set(missing_pos_state.keys()) == missing_pos_snapshot_keys
    assert powerpaint.LEGACY_POSITION_IDS_KEY not in missing_pos_state
    assert len(strict_missing_pos_encoder.load_calls) == 1
    injected_loaded_dict, strict_flag = strict_missing_pos_encoder.load_calls[0]
    assert strict_flag is True
    assert (
        injected_loaded_dict[powerpaint.LEGACY_POSITION_IDS_KEY]
        is strict_missing_pos_encoder.runtime_position_ids
    )
    for i, token in enumerate(powerpaint.PLACEHOLDER_TOKENS):
        assert strict_missing_pos_encoder.emb.trainable_embeddings[
            token
        ].value == float(i + 2)

    strict_missing_encoder = FakeTextEncoder(
        max_positions=77, expect_position_ids_in_state=True
    )
    missing_weight_state = {
        k: v
        for k, v in _base_valid_state().items()
        if k != "text_model.embeddings.token_embedding.wrapped.weight"
    }
    with pytest.raises(ValueError, match="Missing required keys"):
        powerpaint.load_task_text_encoder_state(
            strict_missing_encoder, missing_weight_state, FakeTorch
        )
    assert not strict_missing_encoder.load_calls

    # 3. Invalid legacy position_ids buffers must be rejected before load_state_dict
    for bad_pos_tensor in (
        FakeTensor(list(range(76)), (1, 76), dtype="int64"),
        FakeTensor(list(range(77)), (77,), dtype="int64"),
        FakeTensor(list(range(1, 78)), (1, 77), dtype="int64"),
        FakeTensor([0] * 77, (1, 77), dtype="int64"),
        FakeTensor([float(i) for i in range(77)], (1, 77), dtype="fp32"),
    ):
        bad_encoder = FakeTextEncoder(
            max_positions=77, expect_position_ids_in_state=False
        )
        bad_state = {
            **_base_valid_state(),
            powerpaint.LEGACY_POSITION_IDS_KEY: bad_pos_tensor,
        }
        with pytest.raises(ValueError, match="position_ids"):
            powerpaint.load_task_text_encoder_state(
                bad_encoder, bad_state, FakeTorch
            )
        assert not bad_encoder.load_calls

    # 4. Any other unexpected key must still be rejected (with or without legacy position_ids)
    for extra_state in (
        {
            **_base_valid_state(),
            "text_model.embeddings.unexpected_buffer": FakeTensor(
                1.0, (1, 77)
            ),
        },
        {
            **legacy_state,
            "text_model.embeddings.unexpected_buffer": FakeTensor(
                1.0, (1, 77)
            ),
        },
    ):
        unexpected_encoder = FakeTextEncoder(
            max_positions=77, expect_position_ids_in_state=False
        )
        with pytest.raises(ValueError, match="Unexpected keys"):
            powerpaint.load_task_text_encoder_state(
                unexpected_encoder, extra_state, FakeTorch
            )
        assert not unexpected_encoder.load_calls


def _write_fake_anylora_tree(
    lab_root: Path, in_channels: int = 4, cross_attention_dim: int = 768
):
    anylora_dir = lab_root / "base-models" / "anylora"
    files = {
        "unet/config.json": json.dumps(
            {
                "_class_name": "UNet2DConditionModel",
                "in_channels": in_channels,
                "cross_attention_dim": cross_attention_dim,
            }
        ).encode("utf-8"),
        "unet/diffusion_pytorch_model.fp16.safetensors": b"fake-anylora-unet-fp16-weights",
        "text_encoder/config.json": json.dumps(
            {"architectures": ["CLIPTextModel"], "hidden_size": 768}
        ).encode("utf-8"),
        "text_encoder/model.fp16.safetensors": b"fake-anylora-text-encoder-fp16-weights",
        "vae/config.json": json.dumps(
            {"_class_name": "AutoencoderKL", "latent_channels": 4}
        ).encode("utf-8"),
        "vae/diffusion_pytorch_model.fp16.safetensors": b"fake-anylora-vae-fp16-weights",
        "scheduler/scheduler_config.json": json.dumps(
            {"_class_name": "UniPCMultistepScheduler"}
        ).encode("utf-8"),
        "tokenizer/merges.txt": b"#version: 0.2\n",
        "tokenizer/special_tokens_map.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
        "tokenizer/vocab.json": b"{}",
    }
    for rel, content in files.items():
        path = anylora_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return anylora_dir, files


def test_bases_allowlist_and_preflight_rejects_unknown_base(tmp_path):
    assert set(powerpaint.BASES.keys()) == {"realistic", "anylora"}
    anylora_spec = powerpaint.BASES["anylora"]
    assert anylora_spec["dir"] == powerpaint.LAB / "base-models" / "anylora"
    assert anylora_spec["repository"] == "Lykon/AnyLoRA"
    assert (
        anylora_spec["revision"]
        == "5dac4354b7f5d32f140e1f0fbc849e92b744f347"
    )
    assert (
        anylora_spec["unet_weights_path"]
        == anylora_spec["dir"] / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    )
    assert (
        anylora_spec["text_encoder_weights_path"]
        == anylora_spec["dir"] / "text_encoder" / "model.fp16.safetensors"
    )
    assert anylora_spec["vae_dir"] == anylora_spec["dir"] / "vae"

    for invalid_base in ("sdxl", "../anylora", "/tmp/anylora", "", None):
        with pytest.raises(ValueError, match="Unsupported base model"):
            powerpaint.verify_powerpaint_preflight(tmp_path, base=invalid_base)


def test_verify_anylora_preflight_accepts_valid_synthetic_tree_and_returns_sha256(
    tmp_path,
):
    anylora_dir, files = _write_fake_anylora_tree(tmp_path)

    info = powerpaint.verify_powerpaint_preflight(tmp_path, base="anylora")

    assert info["base_name"] == "anylora"
    assert info["base_repository"] == "Lykon/AnyLoRA"
    assert (
        info["base_revision"]
        == "5dac4354b7f5d32f140e1f0fbc849e92b744f347"
    )
    assert info["base_model_dir"] == anylora_dir
    assert info["unet_config"]["in_channels"] == 4
    assert info["unet_config"]["cross_attention_dim"] == 768

    expected_sha256 = {
        "unet": hashlib.sha256(
            files["unet/diffusion_pytorch_model.fp16.safetensors"]
        ).hexdigest(),
        "text_encoder": hashlib.sha256(
            files["text_encoder/model.fp16.safetensors"]
        ).hexdigest(),
        "vae": hashlib.sha256(
            files["vae/diffusion_pytorch_model.fp16.safetensors"]
        ).hexdigest(),
    }
    assert info["base_weights_sha256"] == expected_sha256


def test_verify_anylora_preflight_rejects_missing_file_and_invalid_unet_channels(
    tmp_path,
):
    anylora_dir, _ = _write_fake_anylora_tree(tmp_path, in_channels=9)
    with pytest.raises(ValueError, match="in_channels"):
        powerpaint.verify_powerpaint_preflight(tmp_path, base="anylora")

    _write_fake_anylora_tree(tmp_path, in_channels=4, cross_attention_dim=1024)
    with pytest.raises(ValueError, match="cross_attention_dim"):
        powerpaint.verify_powerpaint_preflight(tmp_path, base="anylora")

    _write_fake_anylora_tree(tmp_path, in_channels=4, cross_attention_dim=768)
    missing_target = (
        anylora_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    )
    missing_target.unlink()
    with pytest.raises(FileNotFoundError):
        powerpaint.verify_powerpaint_preflight(tmp_path, base="anylora")


def test_verify_anylora_preflight_rejects_symlink_file(tmp_path):
    anylora_dir, _ = _write_fake_anylora_tree(tmp_path)
    outside_target = tmp_path / "external_weights.safetensors"
    outside_target.write_bytes(b"external-weights")
    unet_weights = (
        anylora_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    )
    unet_weights.unlink()
    try:
        unet_weights.symlink_to(outside_target)
    except OSError as exc:
        pytest.skip(f"OS does not allow creating symlinks: {exc}")

    with pytest.raises(ValueError, match="Symlink"):
        powerpaint.verify_powerpaint_preflight(tmp_path, base="anylora")


def _write_fake_lora_run(lab_root: Path, run_name: str = "lora-run-01"):
    run_dir = lab_root / "lora-runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_bytes = b"fake-brushnet-lora-safetensors"
    config_obj = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "init_lora_weights": "gaussian",
        "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    }
    weights_path = run_dir / "brushnet_lora.safetensors"
    config_path = run_dir / "adapter_config.json"
    weights_path.write_bytes(weights_bytes)
    config_path.write_text(json.dumps(config_obj), encoding="utf-8")
    return run_dir, weights_path, config_path, hashlib.sha256(weights_bytes).hexdigest()


def test_verify_lora_preflight_validates_run_name_files_and_symlinks(tmp_path):
    _, weights_path, _, expected_sha = _write_fake_lora_run(
        tmp_path, "lora-run-01"
    )
    info = powerpaint.verify_lora_preflight(tmp_path, "lora-run-01")
    assert info["lora_run"] == "lora-run-01"
    assert info["lora_sha256"] == expected_sha
    assert info["adapter_config"]["r"] == 8

    for bad_name in ("", "../escape", "a/b", "a\\b", "-leading", ".", "..", None):
        with pytest.raises(ValueError, match="Invalid LoRA run name"):
            powerpaint.verify_lora_preflight(tmp_path, bad_name)

    with pytest.raises(FileNotFoundError):
        powerpaint.verify_lora_preflight(tmp_path, "missing-lora")

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    weights_path.unlink()
    try:
        weights_path.symlink_to(outside)
    except OSError:
        pass
    else:
        with pytest.raises(ValueError, match="Symlink"):
            powerpaint.verify_lora_preflight(tmp_path, "lora-run-01")


def test_load_powerpaint_pipeline_offload_false_and_lora_injection_order(
    tmp_path, monkeypatch
):
    import types

    events = []
    _write_fake_lora_run(tmp_path, "lora-run-01")
    (tmp_path / "powerpaint-v2" / "vendor").mkdir(parents=True, exist_ok=True)
    (tmp_path / "powerpaint-v2" / "dependencies").mkdir(
        parents=True, exist_ok=True
    )

    class DummyModule:
        def __init__(self, name):
            self.name = name
            self.dtype = "fp16"

        def to(self, dtype=None, device=None):
            if dtype is not None:
                self.dtype = dtype
            return self

    class DummyPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.brushnet = kwargs["brushnet"]
            self.safety_checker = kwargs.get("safety_checker")
            self.model_cpu_offload_seq = "default"
            self.offload_called = False
            self.vae_slicing_called = False

        def enable_model_cpu_offload(self):
            events.append("enable_model_cpu_offload")
            self.offload_called = True

        def enable_vae_slicing(self):
            events.append("enable_vae_slicing")
            self.vae_slicing_called = True

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoencoderKL = type(
        "AutoencoderKL",
        (),
        {"from_pretrained": staticmethod(lambda *a, **k: DummyModule("vae"))},
    )
    fake_diffusers.UniPCMultistepScheduler = type(
        "UniPCMultistepScheduler",
        (),
        {"from_config": staticmethod(lambda cfg: DummyModule("scheduler"))},
    )

    fake_powerpaint = types.ModuleType("powerpaint")
    fake_models = types.ModuleType("powerpaint.models")
    fake_brushnet_mod = types.ModuleType("powerpaint.models.BrushNet_CA")
    fake_brushnet_mod.BrushNetModel = type(
        "BrushNetModel",
        (),
        {"from_unet": staticmethod(lambda *a, **k: DummyModule("brushnet"))},
    )
    fake_unet_mod = types.ModuleType("powerpaint.models.unet_2d_condition")
    fake_unet_mod.UNet2DConditionModel = type(
        "UNet2DConditionModel",
        (),
        {"from_config": staticmethod(lambda cfg: DummyModule("unet"))},
    )
    fake_pipe_mod = types.ModuleType(
        "powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA"
    )
    fake_pipe_mod.StableDiffusionPowerPaintBrushNetPipeline = DummyPipeline
    fake_utils_mod = types.ModuleType("powerpaint.utils.utils")
    fake_utils_mod.TokenizerWrapper = lambda **k: DummyModule("tokenizer")
    fake_utils_mod.add_tokens = lambda **k: None

    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors_torch = types.ModuleType("safetensors.torch")
    fake_safetensors_torch.load_model = lambda model, path, strict=True: ([], [])
    fake_safetensors_torch.load_file = lambda path: {"lora_A.weight": "tensor"}

    class FakeTextEncoder(DummyModule):
        def __init__(self):
            super().__init__("text_encoder_brushnet")
            emb = types.SimpleNamespace(
                trainable_embeddings={
                    t: types.SimpleNamespace() for t in powerpaint.PLACEHOLDER_TOKENS
                },
                external_embeddings=[
                    {"name": t, "embedding": None}
                    for t in powerpaint.PLACEHOLDER_TOKENS
                ],
            )
            self.text_model = types.SimpleNamespace(
                embeddings=types.SimpleNamespace(token_embedding=emb)
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.CLIPTextModel = type(
        "CLIPTextModel",
        (),
        {"from_pretrained": staticmethod(lambda *a, **k: FakeTextEncoder())},
    )

    fake_peft = types.ModuleType("peft")

    class FakeLoraConfig:
        def __init__(self, r=8, lora_alpha=8, init_lora_weights="gaussian", target_modules=None, **kwargs):
            self.r = r
            self.lora_alpha = lora_alpha
            self.init_lora_weights = init_lora_weights
            self.target_modules = target_modules

    def fake_inject(cfg, model):
        events.append(("inject_adapter_in_model", cfg.r, model.name))

    def fake_set_state(model, state):
        events.append(("set_peft_model_state_dict", model.name, dict(state)))
        return types.SimpleNamespace(unexpected_keys=[])

    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.inject_adapter_in_model = fake_inject
    fake_peft.set_peft_model_state_dict = fake_set_state

    for mod_name, mod_obj in {
        "diffusers": fake_diffusers,
        "powerpaint": fake_powerpaint,
        "powerpaint.models": fake_models,
        "powerpaint.models.BrushNet_CA": fake_brushnet_mod,
        "powerpaint.models.unet_2d_condition": fake_unet_mod,
        "powerpaint.pipelines": types.ModuleType("powerpaint.pipelines"),
        "powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA": fake_pipe_mod,
        "powerpaint.utils": types.ModuleType("powerpaint.utils"),
        "powerpaint.utils.utils": fake_utils_mod,
        "safetensors": fake_safetensors,
        "safetensors.torch": fake_safetensors_torch,
        "transformers": fake_transformers,
        "peft": fake_peft,
    }.items():
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    monkeypatch.setattr(
        powerpaint, "load_task_text_encoder_state", lambda *a, **k: None
    )

    gate = types.SimpleNamespace(
        LAB=tmp_path,
        torch=types.SimpleNamespace(
            float16="fp16",
            load=lambda *a, **k: {},
        ),
    )
    model_info = {
        "weights_dir": tmp_path / "powerpaint-v2" / "weights",
        "base_model_dir": tmp_path / "powerpaint-v2" / "weights" / "base",
        "unet_config": {},
        "scheduler_config": {},
    }

    events.clear()
    pipe_no_offload = powerpaint.load_powerpaint_pipeline(
        gate, model_info=model_info, offload=False
    )
    assert pipe_no_offload.offload_called is False
    assert pipe_no_offload.vae_slicing_called is False
    assert events == []

    events.clear()
    pipe_with_lora = powerpaint.load_powerpaint_pipeline(
        gate, model_info=model_info, offload=True, lora="lora-run-01"
    )
    assert pipe_with_lora.offload_called is True
    assert pipe_with_lora.vae_slicing_called is True
    assert (
        pipe_with_lora.model_cpu_offload_seq
        == powerpaint.MODEL_CPU_OFFLOAD_SEQ
    )
    assert [e[0] if isinstance(e, tuple) else e for e in events] == [
        "inject_adapter_in_model",
        "set_peft_model_state_dict",
        "enable_model_cpu_offload",
        "enable_vae_slicing",
    ]



