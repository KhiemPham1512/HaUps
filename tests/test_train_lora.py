import contextlib
import hashlib
import importlib.util
import io
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageChops

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_CODE = REPO_ROOT / "lab" / "outpainting"

train_spec = importlib.util.spec_from_file_location(
    "train_lora", LAB_CODE / "train_lora.py"
)
train_lora = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train_lora)


def _make_png_bytes(color, size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_import_stays_lightweight_without_torch():
    assert "torch" not in sys.modules
    assert "peft" not in sys.modules


def test_collect_training_images_excludes_sha256_remove_symlink_and_subdirs_without_printing_paths(
    tmp_path, capsys
):
    images_dir = tmp_path / "secret_training_folder"
    images_dir.mkdir(parents=True, exist_ok=True)

    eval_bytes_1 = _make_png_bytes((250, 10, 10), (80, 80))
    eval_bytes_2 = _make_png_bytes((10, 250, 10), (80, 80))
    eval_sha_1 = hashlib.sha256(eval_bytes_1).hexdigest()
    eval_sha_2 = hashlib.sha256(eval_bytes_2).hexdigest()

    manifest_path = tmp_path / "frozen_eval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "anime-extended-evaluation-v2",
                "frozen": True,
                "purpose": "evaluation_only_not_training",
                "files": [
                    {
                        "group": "eval-1",
                        "file": "eval-1.png",
                        "sha256": "a" * 64,
                        "original_file_sha256": eval_sha_1,
                    },
                    {
                        "group": "eval-2",
                        "file": "eval-2.png",
                        "sha256": "b" * 64,
                        "original_file_sha256": eval_sha_2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Excluded by original_file_sha256 match
    (images_dir / "leak_eval_1.png").write_bytes(eval_bytes_1)

    # Excluded because inside a 'remove' directory
    remove_dir = images_dir / "remove"
    remove_dir.mkdir()
    (remove_dir / "discarded.png").write_bytes(
        _make_png_bytes((99, 99, 99), (64, 64))
    )

    # Excluded because non-recursive (nested subdirectory)
    nested_dir = images_dir / "nested_subdir"
    nested_dir.mkdir()
    (nested_dir / "nested.png").write_bytes(
        _make_png_bytes((77, 88, 99), (64, 64))
    )

    # Excluded because non-image extension
    (images_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    # Excluded because symlink
    external_img = tmp_path / "external_source.png"
    external_img.write_bytes(_make_png_bytes((123, 45, 67), (64, 64)))
    symlink_path = images_dir / "symlink_img.png"
    try:
        symlink_path.symlink_to(external_img)
    except OSError:
        pass

    # 9 valid regular files with case-insensitive extensions
    exts = [
        ".png",
        ".PNG",
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".webp",
        ".WEBP",
        ".png",
    ]
    expected_shas = []
    for idx, ext in enumerate(exts):
        raw = _make_png_bytes((idx + 1, idx + 20, idx + 40), (64, 64))
        fname = f"train_secret_{idx:02d}{ext}"
        (images_dir / fname).write_bytes(raw)
        expected_shas.append(hashlib.sha256(raw).hexdigest())

    collected, dataset_id = train_lora.collect_training_images(
        images_dir, manifest_path
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    assert dataset_id == "anime-extended-evaluation-v2"
    assert len(collected) == 9
    got_shas = [sha for _, sha in collected]
    assert got_shas == expected_shas
    assert eval_sha_1 not in got_shas
    assert eval_sha_2 not in got_shas

    # A path containing a directory named 'remove' must skip all files and exit (<8)
    remove_parent_dir = tmp_path / "remove" / "inner_images"
    remove_parent_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(10):
        (remove_parent_dir / f"img_{idx}.png").write_bytes(
            _make_png_bytes((idx, idx, idx))
        )
    with pytest.raises(SystemExit):
        train_lora.collect_training_images(remove_parent_dir, manifest_path)

    # Fewer than 8 valid remaining images must exit without printing any path/filename
    small_dir = tmp_path / "too_few_secret_dir"
    small_dir.mkdir()
    for idx in range(7):
        (small_dir / f"only_{idx}.png").write_bytes(
            _make_png_bytes((idx + 100, 10, 20))
        )
    with pytest.raises(SystemExit) as exc_info:
        train_lora.collect_training_images(small_dir, manifest_path)
    assert "too_few_secret_dir" not in str(exc_info.value)
    assert "only_" not in str(exc_info.value)
    captured_after = capsys.readouterr()
    assert "too_few_secret_dir" not in captured_after.out
    assert "too_few_secret_dir" not in captured_after.err


def test_window_and_mask_geometry_and_overlap_expansion():
    rng = random.Random(42)
    portrait = Image.new("RGB", (600, 900), (80, 140, 200))
    landscape = Image.new("RGB", (1200, 500), (200, 100, 50))

    modes_seen = set()
    flips_seen = set()
    for idx in range(60):
        src = portrait if idx % 2 == 0 else landscape
        ex = train_lora.generate_training_example(src, rng)
        target = ex["target"]
        mask = ex["mask"]
        raw_mask = ex["raw_mask"]
        masked_input = ex["masked_input"]
        x0, y0, x1, y1 = ex["known_box"]
        mode = ex["mode"]
        modes_seen.add(mode)
        flips_seen.add(ex["flipped"])

        assert target.size == (768, 432)
        assert target.mode == "RGB"
        assert mask.size == (768, 432)
        assert mask.mode == "L"
        assert raw_mask.size == (768, 432)
        assert raw_mask.mode == "L"
        assert masked_input.size == (768, 432)
        assert masked_input.mode == "RGB"

        # Verify known_box matches raw_mask (0 = known, 255 = to fill)
        assert ImageChops.invert(raw_mask).getbbox() == (x0, y0, x1, y1)

        if mode == "center_strip":
            strip_w = x1 - x0
            assert int(math.ceil(0.25 * 768)) <= strip_w <= int(0.60 * 768)
            assert y0 == 0 and y1 == 432
            assert abs(x0 - (768 - x1)) <= 1
        elif mode in ("left", "right"):
            keep_w = x1 - x0
            assert int(math.ceil(0.40 * 768)) <= keep_w <= int(0.70 * 768)
            assert y0 == 0 and y1 == 432
            assert (x0 == 0 and mode == "left") or (
                x1 == 768 and mode == "right"
            )
        else:
            keep_h = y1 - y0
            assert int(math.ceil(0.40 * 432)) <= keep_h <= int(0.70 * 432)
            assert x0 == 0 and x1 == 768
            assert (y0 == 0 and mode == "top") or (
                y1 == 432 and mode == "bottom"
            )

        # Verify overlap expansion (pp.OVERLAP_PX = 8) was applied
        px = train_lora.pp.OVERLAP_PX
        exp_x0 = x0 + px if x0 > 0 else x0
        exp_y0 = y0 + px if y0 > 0 else y0
        exp_x1 = x1 - px if x1 < 768 else x1
        exp_y1 = y1 - px if y1 < 432 else y1
        assert ImageChops.invert(mask).getbbox() == (
            exp_x0,
            exp_y0,
            exp_x1,
            exp_y1,
        )

        # Verify masked_input matches prepare_brushnet_input(target, mask)
        expected_masked = train_lora.pp.prepare_brushnet_input(target, mask)
        assert ImageChops.difference(masked_input, expected_masked).getbbox() is None

    assert "center_strip" in modes_seen
    assert len(modes_seen) >= 2
    assert flips_seen == {True, False}


def test_run_name_validation_and_no_overwrite(tmp_path):
    run_dir = train_lora.create_lora_run_dir(tmp_path, "lora-exp-01")
    assert run_dir == tmp_path / "lora-runs" / "lora-exp-01"
    assert run_dir.is_dir()

    # Must never overwrite an existing run directory
    with pytest.raises(FileExistsError):
        train_lora.create_lora_run_dir(tmp_path, "lora-exp-01")

    for invalid_name in (
        "",
        "../escape",
        "sub/dir",
        "sub\\dir",
        "-bad",
        "_bad",
        ".",
        "..",
        "a" * 65,
        None,
    ):
        with pytest.raises(ValueError):
            train_lora.create_lora_run_dir(tmp_path, invalid_name)

    for bad_cli in (
        [
            "../escape",
            "--images-dir",
            "imgs",
            "--exclude-manifest",
            "m.json",
        ],
        [
            "valid-run",
            "--images-dir",
            "imgs",
            "--exclude-manifest",
            "m.json",
            "--steps",
            "0",
        ],
        [
            "valid-run",
            "--images-dir",
            "imgs",
            "--exclude-manifest",
            "m.json",
            "--rank",
            "0",
        ],
    ):
        with pytest.raises(SystemExit):
            train_lora.parse_args(bad_cli)


def _shape_of(nested):
    shape = []
    cur = nested
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return tuple(shape)


def _map_nested(fn, nested):
    if isinstance(nested, list):
        return [_map_nested(fn, item) for item in nested]
    return fn(nested)


class _FakeTensor:
    def __init__(self, data, dtype="float16", device="cuda"):
        self.data = data
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return _shape_of(self.data)

    @property
    def ndim(self):
        return len(self.shape)

    def to(self, device=None, dtype=None):
        return _FakeTensor(
            self.data,
            dtype=dtype or self.dtype,
            device=device or self.device,
        )

    def sum(self, dim):
        assert dim == 1 and self.ndim == 4
        b_len, c_len, h_len, w_len = self.shape
        out = [
            [
                [
                    sum(self.data[b][c][y][x] for c in range(c_len))
                    for x in range(w_len)
                ]
                for y in range(h_len)
            ]
            for b in range(b_len)
        ]
        return _FakeTensor(out, dtype=self.dtype, device=self.device)

    def __getitem__(self, idx):
        if (
            isinstance(idx, tuple)
            and len(idx) == 4
            and idx[1] is None
            and self.ndim == 3
        ):
            return _FakeTensor(
                [[plane] for plane in self.data],
                dtype=self.dtype,
                device=self.device,
            )
        raise IndexError(f"Unsupported index in _FakeTensor: {idx!r}")

    def __lt__(self, other):
        return _FakeTensor(
            _map_nested(lambda v: 1.0 if v < other else 0.0, self.data),
            dtype="bool",
            device=self.device,
        )

    def __mul__(self, other):
        scalar = float(other)
        return _FakeTensor(
            _map_nested(lambda v: v * scalar, self.data),
            dtype=self.dtype,
            device=self.device,
        )

    __rmul__ = __mul__


def test_build_conditioning_mask_convention_matches_pipeline(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.no_grad = contextlib.nullcontext

    def fake_concat(tensors, dim=0):
        assert dim == 1
        b_len = tensors[0].shape[0]
        out = []
        for b in range(b_len):
            channels = []
            for t in tensors:
                channels.extend(t.data[b])
            out.append(channels)
        return _FakeTensor(
            out, dtype=tensors[0].dtype, device=tensors[0].device
        )

    fake_torch.concat = fake_concat

    fake_nn = types.ModuleType("torch.nn")
    fake_F = types.ModuleType("torch.nn.functional")

    def fake_interpolate(tensor, size):
        target_h, target_w = size
        b_len, c_len, src_h, src_w = tensor.shape
        out = [
            [
                [
                    [
                        tensor.data[b][c][(ty * src_h) // target_h][
                            (tx * src_w) // target_w
                        ]
                        for tx in range(target_w)
                    ]
                    for ty in range(target_h)
                ]
                for c in range(c_len)
            ]
            for b in range(b_len)
        ]
        return _FakeTensor(out, dtype=tensor.dtype, device=tensor.device)

    fake_F.interpolate = fake_interpolate
    fake_nn.functional = fake_F
    fake_torch.nn = fake_nn

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_F)

    class FakePipe:
        def __init__(self):
            self.brushnet = types.SimpleNamespace(
                device="cuda", dtype="float16"
            )
            self.vae = types.SimpleNamespace(
                device="cuda",
                dtype="float16",
                config=types.SimpleNamespace(scaling_factor=0.18215),
                encode=self._vae_encode,
            )

        def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            num_images_per_prompt,
            device,
            dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,
        ):
            rgb = image.convert("RGB").resize((width, height))
            chw = [
                [
                    [
                        rgb.getpixel((x, y))[c] / 127.5 - 1.0
                        for x in range(width)
                    ]
                    for y in range(height)
                ]
                for c in range(3)
            ]
            return _FakeTensor([chw], dtype=dtype, device=device)

        def _vae_encode(self, img_tensor):
            b, _, h, w = img_tensor.shape
            latent_data = [
                [
                    [[2.0 for _ in range(w // 8)] for _ in range(h // 8)]
                    for _ in range(4)
                ]
                for _ in range(b)
            ]
            latent = _FakeTensor(
                latent_data,
                dtype=img_tensor.dtype,
                device=img_tensor.device,
            )
            return types.SimpleNamespace(
                latent_dist=types.SimpleNamespace(sample=lambda: latent)
            )

    pipe = FakePipe()
    width, height = 64, 32
    # Toy mask: left half 0 (KNOWN), right half 255 (TO FILL)
    toy_mask = Image.new("L", (width, height), 255)
    toy_mask.paste(0, (0, 0, width // 2, height))
    masked_img = train_lora.pp.prepare_brushnet_input(
        Image.new("RGB", (width, height), (100, 150, 200)), toy_mask
    )

    cond = train_lora.build_conditioning(pipe, masked_img, toy_mask)
    assert cond.shape == (1, 5, height // 8, width // 8)

    # First 4 channels: encoded latents * scaling_factor
    assert all(
        math.isclose(val, 2.0 * 0.18215, rel_tol=1e-5)
        for c in range(4)
        for row in cond.data[0][c]
        for val in row
    )

    # 5th channel: pipeline convention original_mask = (mask.sum(1)[:,None] < 0)
    # 1.0 in known region (left half where mask_pil == 0), 0.0 in unknown region (right half where mask_pil == 255)
    mask_channel = cond.data[0][4]
    latent_half_w = (width // 8) // 2
    assert all(val == 1.0 for row in mask_channel for val in row[:latent_half_w])
    assert all(val == 0.0 for row in mask_channel for val in row[latent_half_w:])


def test_character_mask_preprocesses_like_anime_seg_and_maps_back():
    pytest.importorskip("numpy")
    seen = {}

    class FakeSession:
        def run(self, _outputs, feeds):
            img = feeds["img"]
            seen["shape"], seen["dtype"] = img.shape, str(img.dtype)
            seen["max"] = float(img.max())
            # Character = everything the model sees inside the padded image area
            return [(img.sum(axis=1, keepdims=True) > 0).astype("float32")]

    src = Image.new("RGB", (300, 600), (255, 255, 255))
    mask = train_lora.compute_character_mask(src, FakeSession())
    assert seen["shape"] == (1, 3, 1024, 1024)
    assert seen["dtype"] == "float32"
    assert seen["max"] <= 1.0
    assert mask.mode == "L" and mask.size == (300, 600)
    # Padding is cropped away, so the whole original image maps back as character
    assert mask.getextrema() == (255, 255)


def test_char_mask_yields_background_loss_weight_aligned_with_target():
    rng = random.Random(7)
    src = Image.new("RGB", (900, 600), (0, 0, 255))
    src.paste((255, 0, 0), (0, 0, 450, 600))  # red = character on the left half
    char = Image.new("L", src.size, 0)
    char.paste(255, (0, 0, 450, 600))

    flips = set()
    for _ in range(20):
        ex = train_lora.generate_training_example(src, rng, char_mask=char)
        flips.add(ex["flipped"])
        weight = ex["loss_weight"]
        assert weight.mode == "L" and weight.size == (768, 432)
        tgt = ex["target"].load()
        w = weight.load()
        for x in range(0, 768, 16):
            r, _, b = tgt[x, 216]
            if w[x, 216] == 255:
                assert b > 200 and r < 50  # weighted pixels are background
            if r > 200 and b < 50:
                assert w[x, 216] == 0  # character pixels never weighted
    assert flips == {True, False}


def test_latent_loss_weight_downsamples_to_latent_grid():
    pytest.importorskip("numpy")
    weight = Image.new("L", (768, 432), 0)
    weight.paste(255, (384, 0, 768, 432))
    arr = train_lora.latent_loss_weight(weight)
    assert arr.shape == (54, 96)
    assert float(arr[:, :48].max()) == 0.0
    assert float(arr[:, 48:].min()) == 1.0


def test_parse_args_accepts_seg_model():
    args = train_lora.parse_args(
        ["run-a", "--images-dir", "i", "--exclude-manifest", "m.json", "--seg-model", "s.onnx"]
    )
    assert args.seg_model == "s.onnx"
    args = train_lora.parse_args(["run-a", "--images-dir", "i", "--exclude-manifest", "m.json"])
    assert args.seg_model is None


def test_frame_box_mimics_app_geometry():
    rng = random.Random(3)
    for size in ((600, 900), (900, 1600), (1200, 500), (800, 800)):
        for _ in range(40):
            crop, box = train_lora.sample_frame_box(size, rng)
            cx0, cy0, cx1, cy1 = crop
            x0, y0, x1, y1 = box
            assert 0 <= cx0 < cx1 <= size[0] and 0 <= cy0 < cy1 <= size[1]
            assert 0 <= x0 < x1 <= 768 and 0 <= y0 < y1 <= 432
            bw, bh = x1 - x0, y1 - y0
            assert bh >= 340
            assert 0.25 * 768 - 1 <= bw <= 0.70 * 768 + 1
            # frame is resized into the box without visible distortion
            assert abs((cx1 - cx0) / (cy1 - cy0) - bw / bh) < 0.02


def test_composite_example_generates_only_background_outside_known_box():
    rng = random.Random(11)
    char = Image.new("RGB", (600, 900), (255, 0, 0))
    full = Image.new("L", (600, 900), 255)
    empty = Image.new("L", (600, 900), 0)
    bg = Image.new("RGB", (1600, 900), (0, 255, 0))
    flips = set()
    for idx in range(30):
        mask_in = full if idx % 2 == 0 else empty
        ex = train_lora.generate_composite_example(char, mask_in, bg, rng)
        flips.add(ex["flipped"])
        target, raw_mask = ex["target"], ex["raw_mask"]
        x0, y0, x1, y1 = ex["known_box"]
        assert target.size == (768, 432) and ex["loss_weight"] is None
        assert ImageChops.invert(raw_mask).getbbox() == (x0, y0, x1, y1)
        # everything the model must generate is pure background
        outside = target.copy()
        outside.paste((0, 255, 0), (x0, y0, x1, y1))
        assert outside.getextrema() == ((0, 0), (255, 255), (0, 0))
        inside = target.crop((x0 + 4, y0 + 4, x1 - 4, y1 - 4)).getextrema()
        if idx % 2 == 0:  # character pixels are pasted inside the known box
            assert inside == ((255, 255), (0, 0), (0, 0))
        else:  # no character alpha -> the frame's own background is dropped
            assert inside == ((0, 0), (255, 255), (0, 0))
        expected = train_lora.pp.prepare_brushnet_input(target, ex["mask"])
        assert ImageChops.difference(ex["masked_input"], expected).getbbox() is None
    assert flips == {True, False}


def test_composite_example_plain_background_when_no_background_image():
    rng = random.Random(5)
    char = Image.new("RGB", (600, 900), (255, 0, 0))
    ex = train_lora.generate_composite_example(
        char, Image.new("L", (600, 900), 0), None, rng
    )
    target = ex["target"]
    assert target.size == (768, 432)
    # smooth backdrop: neighbouring pixels differ only slightly
    px = target.load()
    assert all(
        abs(px[x, 200][c] - px[x + 1, 200][c]) <= 3 for x in range(767) for c in range(3)
    )


def test_parse_args_bg_dir_requires_seg_model():
    base = ["run-a", "--images-dir", "i", "--exclude-manifest", "m.json"]
    with pytest.raises(SystemExit):
        train_lora.parse_args(base + ["--bg-dir", "b"])
    args = train_lora.parse_args(base + ["--bg-dir", "b", "--seg-model", "s.onnx"])
    assert args.bg_dir == "b" and args.plain_prob == 0.25
    assert train_lora.parse_args(base + ["--plain-prob", "0"]).plain_prob == 0
