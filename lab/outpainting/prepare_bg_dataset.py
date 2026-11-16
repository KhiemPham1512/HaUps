"""Stream-download RicemanT/Anime-Background-Finetuning-Unprocessed (pinned), keep only
Illustration images tagged "no humans" (and no person tags), double-check with anime-seg,
store a downscaled JPEG copy and delete the original. Screencaps have no tags and are skipped.

CLI:
  python lab/outpainting/prepare_bg_dataset.py [--limit N]

Output: .local-artifacts/outpainting/bg-dataset/{images/*.jpg, manifest.jsonl}. Resumable.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character_check as cc  # noqa: E402

REPO = "RicemanT/Anime-Background-Finetuning-Unprocessed"
REVISION = "9c35b0aeabfb1222c63390b3a3333651ee47c51f"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
MIN_SIZE = (768, 432)
MAX_LONG_EDGE = 1280
MAX_CHARACTER_FRACTION = 0.02
LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
OUT = LAB / "bg-dataset"
TAGS_DIR = OUT / "tags"  # sibling .txt tag files of the dataset, fetched beforehand
PERSON_TAG = re.compile(
    r"\b(\d\+?(girl|boy|other)s?|solo|multiple (girls|boys|others)|people|crowd|"
    r"silhouette|person|human|humans)\b"
)


def is_background_tags(text):
    """True only for 'no humans' tag sets without any other person tag."""
    tags = [t.strip() for t in text.split(",")]
    if "no humans" not in tags:
        return False
    return not any(PERSON_TAG.search(t) for t in tags if t != "no humans")


def character_fraction(mask):
    hist = mask.histogram()
    return sum(hist[128:]) / (mask.width * mask.height)


def judge(image, session):
    """Return (reason, kept_image). reason None means keep."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if min(w, h) < MIN_SIZE[1] or max(w, h) < MIN_SIZE[0]:
        return "too_small", None
    if character_fraction(cc.compute_character_mask(image, session)) > MAX_CHARACTER_FRACTION:
        return "character", None
    scale = MAX_LONG_EDGE / max(w, h)
    if scale < 1:
        image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return None, image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, hf_hub_download

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    download_dir = OUT / "download.tmp"
    manifest_path = OUT / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(line)["path"] for line in manifest_path.read_text(encoding="utf-8").splitlines() if line}
    files = []
    for f in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, recursive=True):
        if not (f.path.lower().endswith(IMAGE_EXTS) and getattr(f, "size", None)):
            continue
        tags = TAGS_DIR / (f.path.rsplit(".", 1)[0] + ".txt")
        if tags.is_file() and is_background_tags(tags.read_text(encoding="utf-8")):
            files.append(f.path)
    files.sort()
    todo = [p for p in files if p not in done][: args.limit]
    print(f"total={len(files)} done={len(done)} todo={len(todo)}", flush=True)
    session = cc.load_session(LAB)

    def fetch(path):
        return path, hf_hub_download(
            REPO, path, repo_type="dataset", revision=REVISION, local_dir=download_dir
        )

    kept = 0
    with ThreadPoolExecutor(4) as pool, manifest_path.open("a", encoding="utf-8") as manifest:
        for i, (path, local) in enumerate(pool.map(fetch, todo), 1):
            local = Path(local)
            raw = local.read_bytes()
            record = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
            try:
                with Image.open(local) as image:
                    reason, out = judge(image, session)
            except Exception as error:  # corrupt/undecodable file
                reason, out = f"decode:{type(error).__name__}", None
            if reason is None:
                name = record["sha256"][:16] + ".jpg"
                out.save(OUT / "images" / name, quality=95)
                record.update(kept=True, file=name, size=list(out.size))
                kept += 1
            else:
                record.update(kept=False, reason=reason)
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            local.unlink(missing_ok=True)
            if i % 100 == 0:
                print(f"{i}/{len(todo)} kept={kept}", flush=True)
    shutil.rmtree(download_dir, ignore_errors=True)
    print(f"finished kept={kept}", flush=True)


if __name__ == "__main__":
    main()
