import importlib.util
from pathlib import Path
import sys
import types

from PIL import Image
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "lab" / "outpainting" / "hires_refine.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hires_refine", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refine_timesteps_properties_and_validation():
    mod = _load_module()

    for strength in (0.35, 0.5, 1.0):
        ts = mod.refine_timesteps(strength, 20)
        assert ts[0] == round(999 * strength)
        assert ts[-1] == 1
        assert len(ts) <= 20
        assert all(isinstance(t, int) for t in ts)
        assert all(a > b for a, b in zip(ts, ts[1:]))

    # Deduplication when steps exceeds available integers
    short_ts = mod.refine_timesteps(0.005, 20)
    assert short_ts[0] == round(999 * 0.005)
    assert short_ts[-1] == 1
    assert len(short_ts) <= 20
    assert all(a > b for a, b in zip(short_ts, short_ts[1:]))

    for bad_strength in (0.0, -0.2, 1.01, float("nan"), True, "0.5"):
        with pytest.raises(ValueError):
            mod.refine_timesteps(bad_strength, 20)

    for bad_steps in (0, -5, 1.5, False, "20"):
        with pytest.raises(ValueError):
            mod.refine_timesteps(0.5, bad_steps)


def test_build_init_composites_source_and_generated_at_hires():
    mod = _load_module()
    source_color = (20, 180, 90)
    generated_color = (210, 40, 140)
    source = Image.new("RGB", (300, 600), source_color)
    generated = Image.new("RGB", (768, 432), generated_color)

    init_rgb, canvas, mask = mod.build_init(source, generated)

    assert init_rgb.size == mod.SIZE == (1536, 864)
    assert canvas.size == mod.SIZE
    assert mask.size == mod.SIZE
    assert init_rgb.mode == "RGB"
    assert canvas.mode == "RGB"
    assert mask.mode == "L"
    assert mask.getextrema() == (0, 255)

    # Pixel well inside the centered known box (768, 432) equals the source color
    assert mask.getpixel((768, 432)) == 0
    assert init_rgb.getpixel((768, 432)) == source_color

    # Pixel in the top-left corner equals the generated background color
    assert mask.getpixel((0, 0)) == 255
    assert init_rgb.getpixel((0, 0)) == generated_color

    with pytest.raises(ValueError):
        mod.build_init(source, Image.new("L", (768, 432), 128))


def test_cli_run_name_validation_rejects_traversal():
    mod = _load_module()

    with pytest.raises(ValueError):
        mod.validate_run_name("../x")

    with pytest.raises(SystemExit):
        mod.parse_args(["../x", "--from-run", "seam-blend-20261116", "--images", "."])

    with pytest.raises(SystemExit):
        mod.parse_args(["valid-run", "--from-run", "../x", "--images", "."])


def test_refine_swaps_and_restores_scheduler(monkeypatch):
    mod = _load_module()

    class FakeTensor:
        def __init__(self, shape, device="cpu", dtype="float32"):
            self.shape = tuple(shape)
            self.device = device
            self.dtype = dtype

        def permute(self, *dims):
            return FakeTensor(tuple(self.shape[d] for d in dims), self.device, self.dtype)

        def unsqueeze(self, dim):
            s = list(self.shape)
            s.insert(dim, 1)
            return FakeTensor(tuple(s), self.device, self.dtype)

        def to(self, device=None, dtype=None):
            if isinstance(device, str) and dtype is None and device in ("float16", "float32"):
                dtype = device
                device = self.device
            return FakeTensor(
                self.shape,
                device if device is not None else self.device,
                dtype if dtype is not None else self.dtype,
            )

        def __mul__(self, other):
            return FakeTensor(self.shape, self.device, self.dtype)

        __rmul__ = __mul__

        def __add__(self, other):
            assert other.shape == self.shape
            return FakeTensor(self.shape, self.device, self.dtype)

    class FakeAlpha:
        def item(self):
            return 0.5

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    class FakeGenerator:
        def __init__(self, device="cpu"):
            self.device = device
            self.seed = None

        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch = types.SimpleNamespace(
        float32="float32",
        float16="float16",
        from_numpy=lambda arr: FakeTensor(arr.shape),
        no_grad=lambda: FakeNoGrad(),
        randn=lambda shape, generator=None, dtype="float32": FakeTensor(shape, "cpu", dtype),
        Generator=FakeGenerator,
        tensor=lambda data, device=None: FakeTensor((len(data),), device=device),
    )

    class FakeArray:
        def __init__(self, shape):
            self.shape = shape

        def __truediv__(self, other):
            return self

        def __sub__(self, other):
            return self

    fake_np = types.ModuleType("numpy")
    fake_np.float32 = "float32"
    fake_np.asarray = lambda img, dtype=None: FakeArray((img.size[1], img.size[0], 3))

    class FakeDPM:
        init_noise_sigma = 1.0

        def __init__(self, config):
            self.config = config
            # refine noises the init with the DDPM forward formula using alphas_cumprod[ts[0]]
            self.alphas_cumprod = {t: FakeAlpha() for t in range(1000)}

        @classmethod
        def from_config(cls, config):
            return cls(config)

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.DPMSolverMultistepScheduler = FakeDPM

    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    original_scheduler = types.SimpleNamespace(config={"beta_start": 0.00085})
    seen = {}

    class FakeVAE:
        dtype = "float16"
        config = types.SimpleNamespace(scaling_factor=0.18215)

        def encode(self, x):
            seen["encoded_shape"] = x.shape
            seen["encoded_device"] = x.device
            seen["encoded_dtype"] = x.dtype
            return types.SimpleNamespace(
                latent_dist=types.SimpleNamespace(
                    mode=lambda: FakeTensor((1, 4, 108, 192), device="cuda", dtype="float16")
                )
            )

    class FakePipe:
        def __init__(self):
            self.vae = FakeVAE()
            self.scheduler = original_scheduler
            self.tiling_enabled = False

        def enable_vae_tiling(self):
            self.tiling_enabled = True

        def __call__(self, **kwargs):
            seen["scheduler_during_call"] = self.scheduler
            seen["kwargs"] = kwargs
            return types.SimpleNamespace(
                images=[Image.new("RGB", mod.SIZE, (50, 60, 70))]
            )

    pipe = FakePipe()
    source = Image.new("RGB", (300, 600), (10, 20, 30))
    generated = Image.new("RGB", (768, 432), (100, 110, 120))

    out = mod.refine(
        pipe,
        fake_torch,
        source,
        generated,
        "anime prompt",
        "neg prompt",
        seed=42,
        strength=0.5,
        steps=20,
    )

    assert out.size == (1536, 864)
    assert pipe.tiling_enabled is True
    assert isinstance(seen["scheduler_during_call"], FakeDPM)
    assert pipe.scheduler is original_scheduler
    assert seen["encoded_shape"] == (1, 3, 864, 1536)
    assert seen["kwargs"]["width"] == 1536
    assert seen["kwargs"]["height"] == 864
    assert seen["kwargs"]["timesteps"] == mod.refine_timesteps(0.5, 20)
    assert seen["kwargs"]["num_inference_steps"] == len(seen["kwargs"]["timesteps"])
