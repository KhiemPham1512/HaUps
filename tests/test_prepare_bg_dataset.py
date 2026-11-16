import importlib.util
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("numpy")

LAB_CODE = Path(__file__).resolve().parents[1] / "lab" / "outpainting"
spec = importlib.util.spec_from_file_location("prepare_bg_dataset", LAB_CODE / "prepare_bg_dataset.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class FakeSession:
    def __init__(self, value):
        self.value = value

    def run(self, _outputs, feeds):
        import numpy as np

        return [np.full((1, 1, 1024, 1024), self.value, dtype=np.float32)]


def test_rejects_small_images():
    reason, out = prep.judge(Image.new("RGB", (700, 400)), FakeSession(0.0))
    assert reason == "too_small" and out is None


def test_rejects_images_with_characters():
    reason, out = prep.judge(Image.new("RGB", (1920, 1080)), FakeSession(1.0))
    assert reason == "character" and out is None


def test_keeps_background_and_downscales_long_edge():
    reason, out = prep.judge(Image.new("RGB", (2560, 1440)), FakeSession(0.0))
    assert reason is None
    assert out.size == (1280, 720) and out.mode == "RGB"


def test_background_tags_require_no_humans_and_no_person_tags():
    assert prep.is_background_tags("no humans, scenery, sky, cloud")
    assert not prep.is_background_tags("1girl, scenery, sky")
    assert not prep.is_background_tags("scenery, sky")
    assert not prep.is_background_tags("no humans, silhouette, city")
    assert not prep.is_background_tags("no humans, 2boys, street")
    assert prep.is_background_tags("no humans, solar panel, roof")  # "solo" must not match "solar"
