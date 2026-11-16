import importlib.util
import random
from pathlib import Path

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location(
    "train_sr", Path(__file__).resolve().parents[1] / "lab" / "upscale" / "train_sr.py"
)
train_sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_sr)


def noisy(size, seed=1):
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    return image


def test_degrade_returns_quarter_size_rgb_and_is_seeded():
    patch = noisy((256, 256))
    a = train_sr.degrade(patch, random.Random(3))
    b = train_sr.degrade(patch, random.Random(3))
    assert a.size == (64, 64) and a.mode == "RGB"
    assert a.tobytes() == b.tobytes()


def test_sample_patch_is_full_size_even_for_small_images():
    for size in ((900, 1400), (300, 200), (256, 256)):
        patch = train_sr.sample_patch(noisy(size), random.Random(0))
        assert patch.size == (256, 256) and patch.mode == "RGB"


def test_sample_patch_prefers_detail_over_flat_area():
    image = Image.new("RGB", (1024, 256), "white")
    image.paste(noisy((256, 256)), (768, 0))
    detailed = sum(
        train_sr.detail_score(train_sr.sample_patch(image, random.Random(seed))) > 5 for seed in range(40)
    )
    assert detailed > 20  # uniform sampling would hit the detailed quarter far less often


def test_list_images_skips_subfolders_and_excluded_hashes(tmp_path):
    noisy((8, 8), 1).save(tmp_path / "a.png")
    noisy((8, 8), 2).save(tmp_path / "b.png")
    (tmp_path / "remove").mkdir()
    noisy((8, 8), 3).save(tmp_path / "remove" / "c.png")
    (tmp_path / "note.txt").write_text("x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"files": [{"sha256": "%s"}]}' % train_sr.sha256_file(tmp_path / "a.png"))
    kept = train_sr.list_images(tmp_path, train_sr.manifest_hashes(manifest))
    assert [p.name for p in kept] == ["b.png"]


def test_parse_args_rejects_bad_run_name():
    base = ["--char-dir", "c", "--bg-dir", "b", "--exclude-manifest", "m"]
    args = train_sr.parse_args(["run-1"] + base)
    assert args.steps == 2000 and args.perceptual == 0.0
    assert train_sr.parse_args(["run-1", "--perceptual", "0.05"] + base).perceptual == 0.05
    with pytest.raises(SystemExit):
        train_sr.parse_args(["../x"] + base)


def test_export_ncnn_renames_pnnx_blobs_for_realesrgan():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab" / "upscale"))
    import export_ncnn

    param = "7767517\n3 3\nInput in0 0 1 in0\nConvolution conv_0 1 1 in0 1 0=64 1=3\nConvolution conv_1 1 1 1 out0 0=3 1=3\n"
    lines = export_ncnn.rename_blobs(param).splitlines()
    assert lines[2] == "Input in0 0 1 data"
    assert lines[3] == "Convolution conv_0 1 1 data 1 0=64 1=3"
    assert lines[4] == "Convolution conv_1 1 1 1 output 0=3 1=3"
    with pytest.raises(SystemExit):
        export_ncnn.rename_blobs("7767517\n1 1\nInput x 0 1 x\n")


def test_fetch_photos_selects_ids_by_exact_keyword():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab" / "upscale"))
    import fetch_photos

    rows = [{"photo_id": "a", "keyword": "Beachwear "}, {"photo_id": "b", "keyword": "beachwear top"},
            {"photo_id": "c", "keyword": "sundress"}, {"photo_id": "a", "keyword": "sundress"}]
    assert fetch_photos.select_ids(rows, ("beachwear", "sundress")) == ["a", "c"]


def test_pick_source_kind_boundaries():
    assert train_sr.pick_source_kind(0.49, 0.15) == "character"
    assert train_sr.pick_source_kind(0.5, 0.15) == "photo"
    assert train_sr.pick_source_kind(0.649, 0.15) == "photo"
    assert train_sr.pick_source_kind(0.65, 0.15) == "background"
    for roll in (0.0, 0.49, 0.5, 0.75, 0.99):
        assert train_sr.pick_source_kind(roll, 0.0) != "photo"


def test_main_validates_photo_args():
    base = ["run-1", "--char-dir", "c", "--bg-dir", "b", "--exclude-manifest", "m"]
    with pytest.raises(SystemExit):
        train_sr.main(base + ["--photo-share", "0.2"])
    with pytest.raises(SystemExit):
        train_sr.main(base + ["--photo-share", "0.6", "--photo-dir", "p"])

