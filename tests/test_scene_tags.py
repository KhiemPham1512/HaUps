import hashlib
import json
import types

from PIL import Image
import pytest

from lab.outpainting import scene_tags


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("star_(sky)", "star"),
        ("spring_(season)", "spring"),
        ("tokyo_(city)", "tokyo"),
        ("cloudy_sky", "cloudy sky"),
        ("east_asian_architecture", "east asian architecture"),
        ("two-tone_background", "two-tone background"),
    ],
)
def test_prompt_text_strips_underscores_and_qualifiers(raw, expected):
    assert scene_tags.prompt_text(raw) == expected


def test_scene_tags_filters_sorts_and_limits_allowlisted_general_tags():
    fake_tag_rows = [
        ("1girl", 0),
        ("hatsune_miku", 4),
        ("outdoors", 0),
        ("star_(sky)", 0),
        ("blue_sky", 0),
        ("forest", 0),
        ("ocean", 4),
    ]
    probs = [[0.99, 0.95, 0.62, 0.85, 0.40, 0.34, 0.91]]

    class FakeSession:
        def run(self, output_names, feed):
            assert output_names is None
            assert "input_1" in feed
            return [probs]

    fake_tagger = types.SimpleNamespace(
        session=FakeSession(),
        input_name="input_1",
        input_size=16,
        tags=fake_tag_rows,
    )
    img = Image.new("RGB", (8, 8), (10, 20, 30))

    result = scene_tags.scene_tags(img, fake_tagger, threshold=0.35, limit=2)
    assert result == [("star_(sky)", 0.85), ("outdoors", 0.62)]

    all_matching = scene_tags.scene_tags(img, fake_tagger)
    assert all_matching == [
        ("star_(sky)", 0.85),
        ("outdoors", 0.62),
        ("blue_sky", 0.40),
    ]
    assert scene_tags.scene_prompt(img, fake_tagger) == "star, outdoors, blue sky"


def test_scene_prompt_returns_empty_string_when_no_tags_pass_threshold():
    class FakeSession:
        def run(self, output_names, feed):
            return [[[0.95, 0.10]]]

    fake_tagger = types.SimpleNamespace(
        session=FakeSession(),
        input_name="input_1",
        input_size=16,
        tags=[("1girl", 0), ("outdoors", 0)],
    )
    img = Image.new("RGB", (8, 8), (10, 20, 30))
    assert scene_tags.scene_tags(img, fake_tagger) == []
    assert scene_tags.scene_prompt(img, fake_tagger) == ""


def test_preprocess_output_shape_dtype_and_bgr_order():
    np = pytest.importorskip("numpy")

    # 2x4 portrait image with pure red (R=255, G=0, B=0)
    image = Image.new("RGB", (2, 4), (255, 0, 0))
    arr = scene_tags.preprocess(image, 4)

    assert arr.shape == (1, 4, 4, 3)
    assert arr.dtype == np.float32
    # Left/right 1 px padding is white (255, 255, 255)
    assert tuple(arr[0, 2, 0]) == (255.0, 255.0, 255.0)
    assert tuple(arr[0, 2, 3]) == (255.0, 255.0, 255.0)
    # Center pixels are BGR for pure red -> (B=0, G=0, R=255)
    assert tuple(arr[0, 2, 1]) == (0.0, 0.0, 255.0)
    assert tuple(arr[0, 2, 2]) == (0.0, 0.0, 255.0)

    # Transparent RGBA pixels paste onto white
    rgba = Image.new("RGBA", (4, 4), (0, 128, 255, 0))
    rgba_arr = scene_tags.preprocess(rgba, 4)
    assert tuple(rgba_arr[0, 0, 0]) == (255.0, 255.0, 255.0)


def test_load_tagger_raises_on_sha_mismatch_or_missing_file(tmp_path):
    tagger_dir = tmp_path / scene_tags.TAGGER_DIR_NAME
    tagger_dir.mkdir()
    model_bytes = b"fake-onnx-model"
    csv_bytes = b"tag_id,name,category,count\n0,outdoors,0,100\n"
    (tagger_dir / "model.onnx").write_bytes(model_bytes)
    (tagger_dir / "selected_tags.csv").write_bytes(csv_bytes)

    good_csv_sha = hashlib.sha256(csv_bytes).hexdigest()
    bad_model_sha = "0" * 64
    (tagger_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "model.onnx": bad_model_sha,
                    "selected_tags.csv": good_csv_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        scene_tags.load_tagger(tmp_path)

    (tagger_dir / "model.onnx").unlink()
    with pytest.raises(RuntimeError, match="Missing WD tagger file"):
        scene_tags.load_tagger(tmp_path)
