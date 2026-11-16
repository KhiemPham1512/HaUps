import json

from PIL import Image
import pytest

from lab.outpainting import scene_prompt_lab


@pytest.mark.parametrize("valid_name", ["scene-tags-20261116", "run_1", "A"])
def test_validate_run_name_accepts_valid_names(valid_name):
    assert scene_prompt_lab.validate_run_name(valid_name) == valid_name


@pytest.mark.parametrize(
    "invalid_name",
    ["", "-bad", "_bad", "bad/name", "bad..name", "a" * 65, None],
)
def test_validate_run_name_rejects_invalid_names(invalid_name):
    with pytest.raises(ValueError, match="invalid run name"):
        scene_prompt_lab.validate_run_name(invalid_name)


def test_tags_only_end_to_end_records_tags_without_gpu(tmp_path, monkeypatch):
    images_dir = tmp_path / "inputs"
    images_dir.mkdir()
    secret_name = "private_photo_name.png"
    Image.new("RGB", (16, 24), (40, 80, 120)).save(
        images_dir / secret_name, format="PNG"
    )

    lab_root = tmp_path / "lab-out"
    lab_root.mkdir()
    monkeypatch.setattr(scene_prompt_lab, "LAB", lab_root)

    sentinel_tagger = object()
    monkeypatch.setattr(
        scene_prompt_lab, "load_tagger", lambda lab: sentinel_tagger
    )
    monkeypatch.setattr(
        scene_prompt_lab,
        "scene_tags",
        lambda image, tagger: [
            ("outdoors", 0.87654),
            ("star_(sky)", 0.54321),
        ],
    )

    summary = scene_prompt_lab.main(
        [
            "tags-only-run",
            "--images",
            str(images_dir),
            "--tags-only",
        ]
    )

    records_path = lab_root / "tags-only-run" / "records.json"
    assert records_path.is_file()
    raw_json = records_path.read_text(encoding="utf-8")
    assert secret_name not in raw_json

    saved = json.loads(raw_json)
    assert saved == summary
    assert saved["run_name"] == "tags-only-run"
    assert saved["tags_only"] is True
    assert len(saved["records"]) == 1
    rec = saved["records"][0]
    assert rec["case"] == "case-01"
    assert rec["tags"] == [
        {"name": "outdoors", "score": 0.877},
        {"name": "star_(sky)", "score": 0.543},
    ]
    assert rec["scene_prompt"] == "outdoors, star"
    assert isinstance(rec["tagger_seconds"], float)
    assert isinstance(rec["total_seconds"], float)
