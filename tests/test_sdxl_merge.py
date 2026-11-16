import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location(
    "sdxl_merge", Path(__file__).resolve().parents[1] / "lab" / "outpainting" / "sdxl_merge.py"
)
sdxl_merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdxl_merge)


def test_same_shape_is_add_difference_in_float32():
    anime, inpaint, base = torch.full((2, 3), 5.0).half(), torch.full((2, 3), 2.0).half(), torch.ones(2, 3).half()
    out = sdxl_merge.merge_tensor(anime, inpaint, base)
    assert out.dtype == torch.float32 and torch.equal(out, torch.full((2, 3), 6.0))


def test_conv_in_adds_anime_delta_to_noise_channels_only():
    anime, base = torch.full((8, 4, 3, 3), 3.0), torch.ones(8, 4, 3, 3)
    inpaint = torch.full((8, 9, 3, 3), 10.0)
    out = sdxl_merge.merge_tensor(anime, inpaint, base)
    assert out.shape == (8, 9, 3, 3)
    assert torch.equal(out[:, :4], torch.full((8, 4, 3, 3), 12.0))
    assert torch.equal(out[:, 4:], inpaint[:, 4:])


def test_unexpected_shape_change_is_rejected():
    with pytest.raises(ValueError):
        sdxl_merge.merge_tensor(torch.ones(4, 4), torch.ones(4, 9), torch.ones(4, 4))
    with pytest.raises(ValueError):
        sdxl_merge.merge_tensor(torch.ones(4, 4), torch.ones(4, 4), torch.ones(4, 5))
