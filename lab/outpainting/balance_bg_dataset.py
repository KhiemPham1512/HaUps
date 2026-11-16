"""Build bg-dataset/images-balanced from bg-dataset/images (run after prepare_bg_dataset.py).

Why: the composite LoRA trained on the raw set collapsed to flat purple/pink fills
(77% of images are sky, about half are blue-toned, 7% interiors). Rules, deterministic:
  1. drop the 20% of images with the least detail (near-flat skies teach flat fills);
  2. cap every mean-hue bucket at HUE_CAP images (most detailed first);
  3. copy images without a "sky" tag twice (streets, interiors, buildings are rare).

CLI: python lab/outpainting/balance_bg_dataset.py
"""

import colorsys
import json
import shutil
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

OUT = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting" / "bg-dataset"
HUE_CAP = 90
DROP_FLATTEST = 0.20
HUES = ("red", "orange", "yellow", "green", "green", "cyan", "cyan", "blue", "blue", "purple", "pink", "red")


def hue_bucket(rgb):
    h, s, _ = colorsys.rgb_to_hsv(*(v / 255 for v in rgb))
    return "grey" if s < 0.12 else HUES[int(h * 12) % 12]


def detail(image):
    small = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    return ImageStat.Stat(small.filter(ImageFilter.FIND_EDGES)).mean[0]


def select(items):
    """items: dicts with file, detail, hue, sky. Returns the file list (repeats = weight)."""
    items = sorted(items, key=lambda i: (-i["detail"], i["file"]))
    items = items[: len(items) - int(len(items) * DROP_FLATTEST)]
    chosen, used = [], {}
    for item in items:
        used[item["hue"]] = used.get(item["hue"], 0) + 1
        if used[item["hue"]] <= HUE_CAP:
            chosen += [item["file"]] * (1 if item["sky"] else 2)
    return chosen


def main():
    items = []
    for line in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line) if line else {}
        if not record.get("kept"):
            continue
        tags = (OUT / "tags" / (record["path"].rsplit(".", 1)[0] + ".txt")).read_text(encoding="utf-8")
        with Image.open(OUT / "images" / record["file"]) as image:
            image = image.convert("RGB")
            mean = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            items.append({
                "file": record["file"],
                "detail": detail(image),
                "hue": hue_bucket(mean),
                "sky": "sky" in {t.strip() for t in tags.split(",")},
            })
    chosen = select(items)
    target = OUT / "images-balanced"
    target.mkdir()  # never overwrite an existing selection
    seen = {}
    for name in chosen:
        seen[name] = seen.get(name, 0) + 1
        shutil.copyfile(OUT / "images" / name, target / f"{Path(name).stem}-{seen[name]}.jpg")
    hues = {}
    for item in items:
        if item["file"] in seen:
            hues[item["hue"]] = hues.get(item["hue"], 0) + 1
    summary = {"source": len(items), "unique": len(seen), "files": len(chosen),
               "no_sky_unique": sum(1 for i in items if i["file"] in seen and not i["sky"]), "hues": hues}
    (OUT / "balanced.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
