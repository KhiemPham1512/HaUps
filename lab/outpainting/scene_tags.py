"""CPU WD14 scene tagger (SmilingWolf/wd-vit-tagger-v3 ONNX) for wallpaper outpainting prompts."""

import csv
import hashlib
import json
from pathlib import Path
import re
import types

from PIL import Image

TAGGER_DIR_NAME = "wd-tagger"
THRESHOLD = 0.35
LIMIT = 8

SCENE_TAGS = frozenset(
    """
    outdoors indoors sky day night cloud cloudy_sky blue_sky night_sky starry_sky
    star_(sky) sunset sunrise evening twilight overcast fog rain snow snowing wind
    beach ocean shore sand water waves ripples reflection splashing underwater lake
    river waterfall pool poolside onsen bathtub tree palm_tree forest bamboo
    bamboo_forest nature plant potted_plant vines leaf falling_leaves autumn_leaves
    flower flower_field cherry_blossoms petals falling_petals rose sunflower
    hydrangea wisteria lotus lily_pad grass field garden park hill mountain
    mountainous_horizon horizon landscape scenery desert cave ice crystal city
    cityscape city_lights tokyo_(city) town village street road alley building
    skyscraper rooftop balcony bridge pier boat ship car train train_station
    train_interior bus_stop power_lines utility_pole lamppost vending_machine sign
    architecture east_asian_architecture shrine temple torii castle church ruins
    stained_glass column pillar stairs fence railing gate door doorway hallway
    window curtains sliding_doors shouji tatami wooden_floor floor ceiling carpet
    rug tiles brick_wall stone_wall room bedroom bed bed_sheet pillow classroom
    school library kitchen bathroom office cafe restaurant shop bookshelf desk
    chair table couch lamp chandelier candle fireplace lantern paper_lantern
    neon_lights stage stage_lights spotlight concert throne moon full_moon
    crescent_moon sun sunlight sunbeam light_rays dappled_sunlight lens_flare bokeh
    light_particles sparkle bubble fireworks rainbow aurora space planet galaxy
    fantasy cyberpunk steampunk summer autumn winter spring_(season)
    simple_background white_background grey_background gradient_background
    blurry_background blue_background pink_background black_background
    yellow_background red_background two-tone_background brown_background
    green_background purple_background orange_background floral_background
    multicolored_background starry_background dark_background sparkle_background
    abstract_background light_blue_background light_brown_background
    leaf_background rainbow_background snowflake_background
    """.split()
)

_PAREN_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REQUIRED_TAGGER_FILES = ("model.onnx", "selected_tags.csv")


def prompt_text(name):
    text = name.replace("_", " ")
    text = _PAREN_QUALIFIER_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_file_sha256(files_entry, filename):
    if isinstance(files_entry, dict):
        val = files_entry.get(filename)
        if isinstance(val, str):
            return val.lower()
        if isinstance(val, dict) and isinstance(val.get("sha256"), str):
            return val["sha256"].lower()
    elif isinstance(files_entry, list):
        for item in files_entry:
            if isinstance(item, dict) and (
                item.get("file") == filename
                or item.get("path") == filename
                or item.get("name") == filename
            ):
                sha = item.get("sha256")
                if isinstance(sha, str):
                    return sha.lower()
    return None


def _verify_tagger_files(tagger_dir):
    manifest_path = tagger_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing WD tagger manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Invalid WD tagger manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or "files" not in manifest:
        raise RuntimeError("WD tagger manifest is missing 'files'")
    files_entry = manifest["files"]
    for filename in _REQUIRED_TAGGER_FILES:
        file_path = tagger_dir / filename
        if not file_path.is_file():
            raise RuntimeError(f"Missing WD tagger file: {file_path}")
        expected_sha = _expected_file_sha256(files_entry, filename)
        if not expected_sha:
            raise RuntimeError(f"Missing sha256 in manifest for {filename}")
        actual_sha = _sha256_file(file_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: {actual_sha} != {expected_sha}"
            )


def _read_selected_tags(csv_path):
    tags = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tags.append((row["name"].strip(), int(row["category"])))
    return tags


def load_tagger(lab_dir):
    tagger_dir = Path(lab_dir) / TAGGER_DIR_NAME
    _verify_tagger_files(tagger_dir)
    tags = _read_selected_tags(tagger_dir / "selected_tags.csv")

    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(tagger_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape
    input_size = int(shape[1])
    return types.SimpleNamespace(
        session=session,
        input_name=input_name,
        input_size=input_size,
        size=input_size,
        tags=tags,
    )


def preprocess(image, size):
    import numpy as np

    if image.mode == "RGBA" or "A" in image.getbands():
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")

    w, h = rgb.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(rgb, ((side - w) // 2, (side - h) // 2))
    if square.size != (size, size):
        square = square.resize((size, size), Image.Resampling.BICUBIC)

    arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(arr[None, ...], dtype=np.float32)


def scene_tags(image, tagger, threshold=THRESHOLD, limit=LIMIT):
    size = getattr(tagger, "input_size", getattr(tagger, "size", 448))
    input_name = getattr(tagger, "input_name", None)
    if input_name is None:
        input_name = tagger.session.get_inputs()[0].name
    try:
        tensor = preprocess(image, size)
    except ModuleNotFoundError:
        tensor = image
    probs = tagger.session.run(None, {input_name: tensor})[0]
    if len(probs) == 1 and hasattr(probs[0], "__len__"):
        probs = probs[0]
    matched = [
        (name, float(score))
        for (name, category), score in zip(tagger.tags, probs)
        if int(category) == 0 and name in SCENE_TAGS and float(score) >= threshold
    ]
    matched.sort(key=lambda item: item[1], reverse=True)
    return matched[:limit]


def scene_prompt(image, tagger):
    return ", ".join(prompt_text(name) for name, _ in scene_tags(image, tagger))
