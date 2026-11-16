import importlib.util
import json
import math
from pathlib import Path

from PIL import Image
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "lab" / "outpainting" / "lowvram_probe.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("lowvram_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_name_validation_rejects_traversal():
    mod = _load_module()

    with pytest.raises(ValueError):
        mod.validate_run_name("../x")

    with pytest.raises(SystemExit):
        mod.parse_args(["upscale", "../x", "--images", "."])

    with pytest.raises(SystemExit):
        mod.parse_args(
            ["outpaint", "../x", "--images", ".", "--case", "1", "--mode", "normal"]
        )

    with pytest.raises(SystemExit):
        mod.parse_args(["compare", "../x", "--case", "1"])


def test_compare_identical_and_known_difference(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "LAB", tmp_path)

    run_dir = tmp_path / "probe-run"
    normal_dir = run_dir / "normal-case-01"
    seq_dir = run_dir / "sequential-case-01"
    normal_dir.mkdir(parents=True)
    seq_dir.mkdir(parents=True)

    base_img = Image.new("RGB", (16, 16), (100, 120, 140))
    base_img.save(normal_dir / "generated.png", format="PNG")
    base_img.save(seq_dir / "generated.png", format="PNG")

    same = mod.main(["compare", "probe-run", "--case", "1"])
    assert same["size_equal"] is True
    assert same["mean_abs_diff"] == 0.0
    assert same["max_abs_diff"] == 0.0
    assert math.isinf(same["psnr"]) and same["psnr"] > 0
    assert same["share_pixels_diff_gt_8"] == 0.0

    saved_same = json.loads(
        (run_dir / "compare-case-01.json").read_text(encoding="utf-8")
    )
    assert saved_same["mean_abs_diff"] == 0.0
    assert math.isinf(saved_same["psnr"])

    diff_img = Image.new("RGB", (16, 16), (112, 120, 140))
    diff_img.save(seq_dir / "generated.png", format="PNG")

    diff = mod.main(["compare", "probe-run", "--case", "1"])
    assert diff["size_equal"] is True
    assert diff["mean_abs_diff"] == pytest.approx(4.0)
    assert diff["max_abs_diff"] == pytest.approx(12.0)
    assert diff["share_pixels_diff_gt_8"] == pytest.approx(1.0)
    expected_mse = (12.0**2) / 3.0
    expected_psnr = 10.0 * math.log10((255.0**2) / expected_mse)
    assert diff["psnr"] == pytest.approx(expected_psnr)


def test_sequential_wrappers_configure_offload():
    mod = _load_module()

    calls = []

    class FakeTextEncoder:
        def __init__(self):
            self.devices = []
            self.device_during_forward = None

        def to(self, device):
            self.devices.append(device)
            return self

        def forward(self, *a, **kw):
            self.device_during_forward = self.devices[-1] if self.devices else None
            return ("out", a, kw)

    class FakePPPipe:
        def __init__(self):
            self.text_encoder = FakeTextEncoder()
            self.text_encoder_brushnet = FakeTextEncoder()

        def enable_sequential_cpu_offload(self):
            calls.append("pp_seq_offload")

        def enable_vae_slicing(self):
            calls.append("pp_vae_slicing")

    seen_pp = {}

    def fake_pp_load(gate, model_info=None, **kw):
        seen_pp["gate"] = gate
        seen_pp["model_info"] = model_info
        seen_pp["kw"] = kw
        calls.append("pp_load")
        return FakePPPipe()

    wrapped_pp = mod.wrap_powerpaint_sequential(fake_pp_load)
    pipe_pp = wrapped_pp("gate-obj", "info-obj", offload=True)
    assert isinstance(pipe_pp, FakePPPipe)
    assert seen_pp["gate"] == "gate-obj"
    assert seen_pp["model_info"] == "info-obj"
    assert seen_pp["kw"]["offload"] is False
    assert calls == ["pp_load", "pp_seq_offload", "pp_vae_slicing"]
    assert "text_encoder" in pipe_pp._exclude_from_cpu_offload
    assert "text_encoder_brushnet" in pipe_pp._exclude_from_cpu_offload
    assert pipe_pp.text_encoder.devices == ["cpu"]
    out = pipe_pp.text_encoder.forward("tokens", flag=1)
    assert out == ("out", ("tokens",), {"flag": 1})
    assert pipe_pp.text_encoder.device_during_forward == "cuda"
    assert pipe_pp.text_encoder.devices == ["cpu", "cuda", "cpu"]
    assert pipe_pp.text_encoder.devices[-1] == "cpu"

    sx_calls = []

    class FakeSDXLPipe:
        def remove_all_hooks(self):
            sx_calls.append("sx_remove_hooks")

        def enable_sequential_cpu_offload(self):
            sx_calls.append("sx_seq_offload")

    def fake_sx_load(*a, **kw):
        sx_calls.append(("sx_load", a, kw))
        return FakeSDXLPipe()

    wrapped_sx = mod.wrap_sdxl_sequential(fake_sx_load)
    pipe_sx = wrapped_sx("torch-mod", unet_dir="u")
    assert isinstance(pipe_sx, FakeSDXLPipe)
    assert sx_calls == [
        ("sx_load", ("torch-mod",), {"unet_dir": "u"}),
        "sx_remove_hooks",
        "sx_seq_offload",
    ]


def test_peak_ram_mb_returns_number_or_none():
    mod = _load_module()
    val = mod.peak_ram_mb()
    assert val is None or (
        isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0
    )
