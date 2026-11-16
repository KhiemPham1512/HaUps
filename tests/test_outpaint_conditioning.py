import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

spec = importlib.util.spec_from_file_location(
    "conditioning_gpu",
    Path(__file__).resolve().parents[1]
    / "lab/outpainting/conditioning_gpu.py",
)
conditioning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conditioning)

fixture_spec = importlib.util.spec_from_file_location(
    "make_fixture",
    Path(__file__).resolve().parents[1]
    / "lab/outpainting/make_fixture.py",
)
make_fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(make_fixture)


def test_import_stays_lightweight_without_torch_or_diffusers():
    assert "torch" not in sys.modules
    assert "diffusers" not in sys.modules


def test_portrait_reference_square_preserves_marked_edge_pixels_and_source():
    source = Image.new("RGB", (81, 109), "#193d71")
    edge_marks = {
        (0, 0): (255, 10, 20),
        (80, 0): (10, 255, 30),
        (0, 108): (40, 50, 255),
        (80, 108): (255, 220, 15),
        (40, 0): (200, 100, 50),
        (40, 108): (50, 200, 180),
        (0, 54): (180, 40, 210),
        (80, 54): (90, 240, 110),
    }
    for pos, color in edge_marks.items():
        source.putpixel(pos, color)
    before = source.tobytes()

    square = conditioning.reference_image(source)

    assert square.mode == "RGB"
    assert square.size == (109, 109)
    assert square.getpixel((0, 0)) == (127, 127, 127)
    assert square.getpixel((13, 54)) == (127, 127, 127)
    assert square.getpixel((95, 54)) == (127, 127, 127)
    assert square.getpixel((108, 108)) == (127, 127, 127)
    for (x, y), color in edge_marks.items():
        assert square.getpixel((x + 14, y)) == color
    assert ImageChops.difference(square.crop((14, 0, 95, 109)), source).getbbox() is None
    square.putpixel((14, 0), (0, 0, 0))
    assert source.tobytes() == before


@pytest.mark.parametrize(
    ("size", "expected_box", "pad_probe"),
    [
        ((211, 83), (0, 64, 211, 147), (105, 0)),
        ((101, 101), (0, 0, 101, 101), None),
        ((17, 219), (101, 0, 118, 219), (0, 109)),
    ],
)
def test_wide_and_varied_reference_square_preserves_observed_pixels(
    size, expected_box, pad_probe
):
    source = Image.new("RGB", size, "#224466")
    corners = {
        (0, 0): (250, 30, 40),
        (size[0] - 1, 0): (30, 250, 40),
        (0, size[1] - 1): (40, 30, 250),
        (size[0] - 1, size[1] - 1): (250, 240, 30),
    }
    for pos, color in corners.items():
        source.putpixel(pos, color)
    before = source.tobytes()

    square = conditioning.reference_image(source)

    side = max(size)
    assert square.mode == "RGB"
    assert square.size == (side, side)
    if pad_probe is not None:
        assert square.getpixel(pad_probe) == (127, 127, 127)
    assert ImageChops.difference(square.crop(expected_box), source).getbbox() is None
    assert source.tobytes() == before


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_reference_image_rejects_non_rgb_mode(mode):
    with pytest.raises(ValueError):
        conditioning.reference_image(Image.new(mode, (32, 48)))


def test_new_output_creates_fresh_and_refuses_traversal_and_overwrite(tmp_path):
    created = conditioning.new_output(tmp_path, "anime_cond-01")
    assert created == tmp_path / "anime_cond-01"
    assert created.is_dir()

    for name in ("../escape", "a/b", ".", "..", "", "a\\b", "-bad", "a" * 65):
        with pytest.raises(ValueError):
            conditioning.new_output(tmp_path, name)

    with pytest.raises(FileExistsError):
        conditioning.new_output(tmp_path, "anime_cond-01")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["anime_cond-01"]


def test_parse_args_defaults_to_sd15_and_rejects_arbitrary_models():
    default_args = conditioning.parse_args(["smoke-run"])
    assert default_args.run_name == "smoke-run"
    assert default_args.model == "sd15"
    assert default_args.sampler == "ddim"
    assert default_args.dataset == "development"
    assert default_args.recipes == "all"
    assert default_args.full is False

    for model_name in ("sd15", "dreamshaper8", "anylora"):
        for sampler in ("ddim", "deis"):
            for dataset in ("development", "evaluation", "extended-evaluation-v2"):
                for recipes in ("all", "baseline"):
                    parsed = conditioning.parse_args(
                        [
                            "full-run",
                            "--model",
                            model_name,
                            "--sampler",
                            sampler,
                            "--dataset",
                            dataset,
                            "--recipes",
                            recipes,
                            "--full",
                        ]
                    )
                    assert parsed.run_name == "full-run"
                    assert parsed.model == model_name
                    assert parsed.sampler == sampler
                    assert parsed.dataset == dataset
                    assert parsed.recipes == recipes
                    assert parsed.full is True

    for invalid_model in (
        "../other-model",
        "C:/models/custom",
        "Linaqruf/anything-v3.0",
        "sdxl",
    ):
        with pytest.raises(SystemExit):
            conditioning.parse_args(["bad-run", "--model", invalid_model])

    for invalid_sampler in ("euler", "pndm", "../deis", "DDIMScheduler"):
        with pytest.raises(SystemExit):
            conditioning.parse_args(["bad-run", "--sampler", invalid_sampler])

    for invalid_dataset in (
        "train",
        "holdout",
        "extended-evaluation",
        "../evaluation",
        "../extended-evaluation-v2",
        "evaluation-anime-extended-20261114-v2",
        "C:/images",
    ):
        with pytest.raises(SystemExit):
            conditioning.parse_args(["bad-run", "--dataset", invalid_dataset])

    for invalid_recipes in ("ip-adapter-0.5", "adapter", "../baseline", "none"):
        with pytest.raises(SystemExit):
            conditioning.parse_args(["bad-run", "--recipes", invalid_recipes])


def _setup_synthetic_lab(lab_root: Path):
    scheduler_dir = lab_root / "model" / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    scheduler_bytes = json.dumps(
        {
            "_class_name": "DDIMScheduler",
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
        }
    ).encode("utf-8")
    (scheduler_dir / "scheduler_config.json").write_bytes(scheduler_bytes)

    sd15_unet_dir = lab_root / "model" / "unet"
    sd15_unet_dir.mkdir(parents=True, exist_ok=True)
    (sd15_unet_dir / "config.json").write_text(
        json.dumps({"in_channels": 9}), encoding="utf-8"
    )
    (lab_root / "manifest.json").write_text(
        json.dumps(
            {
                "revision": "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
            }
        ),
        encoding="utf-8",
    )

    candidates = (
        (
            "dreamshaper8",
            "Lykon/dreamshaper-8-inpainting",
            "a09e9d0bb9849aaf46c8fc59dfa604796da15a39",
            9,
            "DEISMultistepScheduler",
        ),
        (
            "anylora",
            "Lykon/AnyLoRA",
            "5dac4354b7f5d32f140e1f0fbc849e92b744f347",
            4,
            "PNDMScheduler",
        ),
    )
    for name, repo, revision, in_channels, published_scheduler in candidates:
        model_dir = lab_root / "candidate-models" / name
        (model_dir / "unet").mkdir(parents=True, exist_ok=True)
        (model_dir / "unet" / "config.json").write_text(
            json.dumps({"in_channels": in_channels}), encoding="utf-8"
        )
        files_manifest = []
        for rel_path in conditioning.CORE_FP16_SAFETENSORS:
            target = model_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{name}:{rel_path}".encode("utf-8")
            target.write_bytes(payload)
            files_manifest.append(
                {
                    "file": rel_path,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        cand_sched_rel = "scheduler/scheduler_config.json"
        cand_sched_path = model_dir / cand_sched_rel
        cand_sched_path.parent.mkdir(parents=True, exist_ok=True)
        cand_sched_bytes = json.dumps(
            {
                "_class_name": published_scheduler,
                "beta_schedule": "scaled_linear",
                "num_train_timesteps": 1000,
                "solver_order": 2,
                "model_tag": name,
            }
        ).encode("utf-8")
        cand_sched_path.write_bytes(cand_sched_bytes)
        files_manifest.append(
            {
                "file": cand_sched_rel,
                "bytes": len(cand_sched_bytes),
                "sha256": hashlib.sha256(cand_sched_bytes).hexdigest(),
            }
        )
        manifest_payload = {
            "complete": True,
            "repo": repo,
            "revision": revision,
            "files": files_manifest,
        }
        (model_dir / "manifest.json").write_text(
            "\ufeff" + json.dumps(manifest_payload), encoding="utf-8"
        )


@pytest.mark.parametrize("sampler", ["ddim", "deis"])
@pytest.mark.parametrize(
    ("model_name", "expected_rel_dir", "expected_channels", "expected_mode"),
    [
        ("sd15", Path("model"), 9, "native9"),
        ("dreamshaper8", Path("candidate-models/dreamshaper8"), 9, "native9"),
        ("anylora", Path("candidate-models/anylora"), 4, "latent-mask-blend4"),
    ],
)
def test_load_base_pipeline_uses_selected_model_and_fixed_ddim_without_safety_checker(
    tmp_path,
    sampler,
    model_name,
    expected_rel_dir,
    expected_channels,
    expected_mode,
):
    _setup_synthetic_lab(tmp_path)
    captured = {}

    class FakeDDIMScheduler:
        def __init__(self, config):
            self.config = dict(config)

        @classmethod
        def from_config(cls, config):
            captured["scheduler_factory"] = "DDIMScheduler"
            captured["scheduler_config"] = dict(config)
            return cls(config)

    class FakeDEISMultistepScheduler:
        def __init__(self, config):
            self.config = dict(config)

        @classmethod
        def from_config(cls, config):
            captured["scheduler_factory"] = "DEISMultistepScheduler"
            captured["scheduler_config"] = dict(config)
            return cls(config)

    class FakePipelineFactory:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            captured["model_path"] = Path(model_path)
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                safety_checker=kwargs.get("safety_checker"),
                unet=SimpleNamespace(
                    config=SimpleNamespace(in_channels=expected_channels)
                ),
                scheduler=SimpleNamespace(_class_name="PNDMScheduler"),
            )

    fake_gate = SimpleNamespace(
        LAB=tmp_path,
        torch=SimpleNamespace(float16="fake_torch_float16"),
        diffusers=SimpleNamespace(
            DDIMScheduler=FakeDDIMScheduler,
            DEISMultistepScheduler=FakeDEISMultistepScheduler,
        ),
        StableDiffusionInpaintPipeline=FakePipelineFactory,
    )

    base_info = conditioning.verify_base_model_preflight(
        tmp_path, model_name, sampler=sampler
    )
    pipe = conditioning.load_base_pipeline(
        fake_gate, model_name, base_info=base_info, sampler=sampler
    )

    assert captured["model_path"] == tmp_path / expected_rel_dir
    assert captured["kwargs"] == {
        "safety_checker": None,
        "feature_extractor": None,
        "requires_safety_checker": False,
        "torch_dtype": "fake_torch_float16",
        "variant": "fp16",
        "use_safetensors": True,
        "local_files_only": True,
    }
    assert pipe.safety_checker is None
    assert pipe.unet.config.in_channels == expected_channels
    assert base_info["model_name"] == model_name
    assert base_info["in_channels"] == expected_channels
    assert base_info["inpaint_mode"] == expected_mode
    assert base_info["selected_sampler"] == sampler

    if sampler == "ddim":
        expected_sched_path = tmp_path / "model" / "scheduler" / "scheduler_config.json"
        assert isinstance(pipe.scheduler, FakeDDIMScheduler)
        assert captured["scheduler_factory"] == "DDIMScheduler"
        assert captured["scheduler_config"]["_class_name"] == "DDIMScheduler"
        assert base_info["scheduler_class"] == "DDIMScheduler"
    else:
        expected_sched_path = (
            tmp_path / expected_rel_dir / "scheduler" / "scheduler_config.json"
        )
        assert isinstance(pipe.scheduler, FakeDEISMultistepScheduler)
        assert captured["scheduler_factory"] == "DEISMultistepScheduler"
        assert base_info["scheduler_class"] == "DEISMultistepScheduler"
        if model_name != "sd15":
            assert captured["scheduler_config"]["model_tag"] == model_name
            assert (
                base_info["scheduler_config_sha256"]
                != conditioning.sha256_file(
                    tmp_path / "model" / "scheduler" / "scheduler_config.json"
                )
            )

    assert base_info["scheduler_config_sha256"] == conditioning.sha256_file(
        expected_sched_path
    )
    assert captured["scheduler_config"] == base_info["scheduler_config"]
    if model_name != "sd15":
        for core_file in conditioning.CORE_FP16_SAFETENSORS:
            assert core_file in base_info["hashes"]

    with pytest.raises(ValueError, match="sampler"):
        conditioning.verify_base_model_preflight(tmp_path, model_name, sampler="pndm")


def test_verify_base_model_preflight_rejects_manifest_mismatch_and_escaping_file(
    tmp_path,
):
    _setup_synthetic_lab(tmp_path)
    dream_dir = tmp_path / "candidate-models" / "dreamshaper8"
    manifest_path = dream_dir / "manifest.json"
    valid_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    escaped_manifest = dict(valid_manifest)
    escaped_manifest["files"] = list(valid_manifest["files"]) + [
        {
            "file": "../outside.safetensors",
            "bytes": 4,
            "sha256": hashlib.sha256(b"test").hexdigest(),
        }
    ]
    (tmp_path / "candidate-models" / "outside.safetensors").write_bytes(b"test")
    manifest_path.write_text(json.dumps(escaped_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        conditioning.verify_base_model_preflight(tmp_path, "dreamshaper8")

    bad_hash_manifest = dict(valid_manifest)
    bad_hash_manifest["files"] = [dict(entry) for entry in valid_manifest["files"]]
    bad_hash_manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(bad_hash_manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        conditioning.verify_base_model_preflight(tmp_path, "dreamshaper8")

    missing_core_manifest = dict(valid_manifest)
    missing_core_manifest["files"] = [
        entry
        for entry in valid_manifest["files"]
        if entry["file"] != "vae/diffusion_pytorch_model.fp16.safetensors"
    ]
    manifest_path.write_text(json.dumps(missing_core_manifest), encoding="utf-8")
    with pytest.raises(AssertionError, match="core fp16 safetensors"):
        conditioning.verify_base_model_preflight(tmp_path, "dreamshaper8")


def _setup_synthetic_evaluation_dataset(
    lab_root: Path,
    dir_name: str = conditioning.EVALUATION_DIR_NAME,
    dataset_id: str = conditioning.EVALUATION_DATASET_ID,
    groups: tuple[str, ...] = ("eval-a", "eval-b"),
):
    eval_dir = lab_root / dir_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, group in enumerate(groups):
        img = Image.new("RGB", (64 + idx * 8, 96), (30 + idx * 20, 80, 140))
        file_name = f"{group}.png"
        file_path = eval_dir / file_name
        img.save(file_path, format="PNG")
        records.append(
            {
                "group": group,
                "file": file_name,
                "width": img.width,
                "height": img.height,
                "sha256": conditioning.sha256_file(file_path),
                "source_rgb_sha256": hashlib.sha256(img.tobytes()).hexdigest(),
                "original_file_sha256": hashlib.sha256(
                    f"orig-{group}".encode("utf-8")
                ).hexdigest(),
            }
        )
    manifest = {
        "dataset": dataset_id,
        "frozen": True,
        "purpose": "evaluation_only_not_training",
        "blind_holdout": False,
        "selected_at_utc": "2026-11-12T16:30:00Z",
        "selection_note": (
            "Synthetic anime fixture samples for unit test validation."
        ),
        "files": records,
    }
    manifest_path = eval_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return eval_dir, manifest_path, manifest


def test_evaluation_sources_validates_frozen_manifest_and_rejects_invalid(tmp_path):
    eval_dir, manifest_path, valid_manifest = _setup_synthetic_evaluation_dataset(
        tmp_path
    )

    sources, manifest_sha = conditioning.evaluation_sources(tmp_path)
    assert list(sources.keys()) == ["eval-a", "eval-b"]
    assert sources["eval-a"] == (eval_dir / "eval-a.png").resolve()
    assert sources["eval-b"] == (eval_dir / "eval-b.png").resolve()
    assert manifest_sha == conditioning.sha256_file(manifest_path)

    unfrozen = dict(valid_manifest, frozen=False)
    manifest_path.write_text(json.dumps(unfrozen), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        conditioning.evaluation_sources(tmp_path)

    wrong_dataset = dict(valid_manifest, dataset="other-dataset")
    manifest_path.write_text(json.dumps(wrong_dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset"):
        conditioning.evaluation_sources(tmp_path)

    traversal = dict(valid_manifest)
    traversal["files"] = [
        dict(valid_manifest["files"][0], file="../eval-a.png"),
        dict(valid_manifest["files"][1]),
    ]
    manifest_path.write_text(json.dumps(traversal), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        conditioning.evaluation_sources(tmp_path)

    tampered = dict(valid_manifest)
    tampered["files"] = [
        dict(valid_manifest["files"][0], sha256="0" * 64),
        dict(valid_manifest["files"][1]),
    ]
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        conditioning.evaluation_sources(tmp_path)


def test_extended_evaluation_v2_sources_validates_8_group_manifest_and_rejects_invalid(
    tmp_path,
):
    parsed = conditioning.parse_args(
        ["extended-run", "--dataset", "extended-evaluation-v2", "--recipes", "baseline"]
    )
    assert parsed.dataset == "extended-evaluation-v2"
    assert parsed.recipes == "baseline"

    expected_groups = tuple(f"eval-{i}" for i in range(1, 9))
    extended_dir, manifest_path, valid_manifest = _setup_synthetic_evaluation_dataset(
        tmp_path,
        dir_name="evaluation-anime-extended-20261114-v2",
        dataset_id="anime-extended-evaluation-v2",
        groups=expected_groups,
    )

    sources, manifest_sha = conditioning.evaluation_sources(
        tmp_path, dataset="extended-evaluation-v2"
    )
    assert list(sources.keys()) == list(expected_groups)
    for group in expected_groups:
        assert sources[group] == (extended_dir / f"{group}.png").resolve()
    assert manifest_sha == conditioning.sha256_file(manifest_path)

    for bad_dataset in (
        "development",
        "extended-evaluation",
        "../evaluation-anime-extended-20261114-v2",
        "evaluation-anime-extended-20261114-v2",
        "C:/images/extended",
    ):
        with pytest.raises(ValueError, match="dataset"):
            conditioning.evaluation_sources(tmp_path, dataset=bad_dataset)

    unfrozen = dict(valid_manifest, frozen=False)
    manifest_path.write_text(json.dumps(unfrozen), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen"):
        conditioning.evaluation_sources(tmp_path, dataset="extended-evaluation-v2")

    wrong_id = dict(valid_manifest, dataset="anime-extended-evaluation-v1")
    manifest_path.write_text(json.dumps(wrong_id), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset"):
        conditioning.evaluation_sources(tmp_path, dataset="extended-evaluation-v2")

    missing_group = dict(valid_manifest, files=valid_manifest["files"][:-1])
    manifest_path.write_text(json.dumps(missing_group), encoding="utf-8")
    with pytest.raises(ValueError, match="8 files"):
        conditioning.evaluation_sources(tmp_path, dataset="extended-evaluation-v2")

    traversal = dict(valid_manifest)
    traversal["files"] = [
        dict(valid_manifest["files"][0], file="../eval-1.png"),
        *[dict(entry) for entry in valid_manifest["files"][1:]],
    ]
    manifest_path.write_text(json.dumps(traversal), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        conditioning.evaluation_sources(tmp_path, dataset="extended-evaluation-v2")

    tampered = dict(valid_manifest)
    tampered["files"] = [
        dict(valid_manifest["files"][0], sha256="0" * 64),
        *[dict(entry) for entry in valid_manifest["files"][1:]],
    ]
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        conditioning.evaluation_sources(tmp_path, dataset="extended-evaluation-v2")


def test_build_fixture_creates_verified_manifest_without_leaking_source_names(
    tmp_path, monkeypatch, capsys
):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    src_dir = tmp_path / "private_pool"
    src_dir.mkdir()

    sources = []
    orig_hashes = []
    for i in range(1, 9):
        secret_path = src_dir / f"secret_candidate_name_{i:02d}.png"
        img = Image.new("RGB", (40 + i * 4, 52), (10 * i, 90, 160))
        img.save(secret_path, format="PNG")
        sources.append(secret_path)
        orig_hashes.append(hashlib.sha256(secret_path.read_bytes()).hexdigest())

    list_file = tmp_path / "sources.txt"
    list_file.write_text(
        "\n\n".join(str(p) for p in sources) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(make_fixture, "LAB", lab_dir)
    make_fixture.main(
        [
            "--dataset-id",
            "anime-extended-evaluation-v2",
            "--out",
            "evaluation-anime-extended-20261114-v2",
            "--sources",
            str(list_file),
            "--note",
            "8 synthetic evaluation images",
        ]
    )
    captured = capsys.readouterr()
    out_dir = lab_dir / "evaluation-anime-extended-20261114-v2"
    manifest_raw = (out_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_raw)

    for src_path, orig_sha in zip(sources, orig_hashes):
        assert src_path.name not in manifest_raw
        assert str(src_dir) not in manifest_raw
        assert src_path.name not in captured.out
        assert src_path.name not in captured.err
        assert str(src_dir) not in captured.out
        assert str(src_dir) not in captured.err
        assert hashlib.sha256(src_path.read_bytes()).hexdigest() == orig_sha

    assert manifest["dataset"] == "anime-extended-evaluation-v2"
    assert manifest["frozen"] is True
    assert manifest["purpose"] == "evaluation_only_not_training"
    assert manifest["blind_holdout"] is False
    assert manifest["selection_note"] == "8 synthetic evaluation images"
    assert len(manifest["files"]) == 8

    loaded_sources, manifest_sha = conditioning.evaluation_sources(
        lab_dir, dataset="extended-evaluation-v2"
    )
    assert list(loaded_sources.keys()) == [f"eval-{i}" for i in range(1, 9)]
    assert manifest_sha == conditioning.sha256_file(out_dir / "manifest.json")


def test_build_fixture_rejects_existing_out_and_remove_paths_and_applies_exif(
    tmp_path, monkeypatch
):
    lab_dir = tmp_path / "lab"
    lab_dir.mkdir()
    good_dir = tmp_path / "good_sources"
    good_dir.mkdir()

    raw_img = Image.new("RGB", (48, 24), (20, 40, 60))
    raw_img.putpixel((0, 0), (250, 15, 25))
    exif = Image.Exif()
    exif[0x0112] = 6  # Rotate 270 CW -> transposed size (24, 48)
    oriented_src = good_dir / "oriented_sample.png"
    raw_img.save(oriented_src, format="PNG", exif=exif.tobytes())
    raw_file_sha = hashlib.sha256(oriented_src.read_bytes()).hexdigest()
    expected_oriented = Image.open(oriented_src)
    from PIL import ImageOps

    expected_rgb = ImageOps.exif_transpose(expected_oriented).convert("RGB")
    expected_rgb_sha = hashlib.sha256(expected_rgb.tobytes()).hexdigest()

    out_dir = make_fixture.build_fixture(
        lab_dir,
        dataset_id="custom-3-v1",
        out_name="custom-fixture-v1",
        sources=[oriented_src],
        note="EXIF orientation check",
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"][0]
    assert entry["group"] == "eval-1"
    assert entry["file"] == "eval-1.png"
    assert (entry["width"], entry["height"]) == (24, 48)
    assert entry["original_file_sha256"] == raw_file_sha
    assert entry["source_rgb_sha256"] == expected_rgb_sha
    with Image.open(out_dir / "eval-1.png") as saved_png:
        assert saved_png.size == (24, 48)
        assert hashlib.sha256(saved_png.tobytes()).hexdigest() == expected_rgb_sha

    monkeypatch.setitem(
        conditioning.EVALUATION_DATASETS,
        "custom-test",
        ("custom-fixture-v1", "custom-3-v1", ("eval-1",)),
    )
    verified, _ = conditioning.evaluation_sources(lab_dir, dataset="custom-test")
    assert list(verified.keys()) == ["eval-1"]

    with pytest.raises(FileExistsError):
        make_fixture.build_fixture(
            lab_dir,
            dataset_id="custom-3-v1",
            out_name="custom-fixture-v1",
            sources=[oriented_src],
            note="Should refuse overwrite",
        )

    for bad_out in ("../escape", "sub/dir", "sub\\dir", "..", ".", ""):
        with pytest.raises(ValueError):
            make_fixture.build_fixture(
                lab_dir,
                dataset_id="custom-3-v1",
                out_name=bad_out,
                sources=[oriented_src],
                note="Bad out",
            )

    remove_dir = tmp_path / "Remove" / "nested"
    remove_dir.mkdir(parents=True)
    removed_img = remove_dir / "candidate.png"
    Image.new("RGB", (32, 32), "blue").save(removed_img, format="PNG")
    with pytest.raises(ValueError, match="remove"):
        make_fixture.build_fixture(
            lab_dir,
            dataset_id="custom-3-v1",
            out_name="should-not-exist",
            sources=[removed_img],
            note="Must reject remove folder",
        )
    assert not (lab_dir / "should-not-exist").exists()



