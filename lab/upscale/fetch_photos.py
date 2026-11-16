"""Download Unsplash Lite photos matching keywords (for SR training data; Lite TERMS allow internal ML training).

CLI: python lab/upscale/fetch_photos.py --index DIR --out DIR [--keyword beachwear ...] [--size 1920]
Writes OUT/<photo_id>.jpg and OUT/manifest.json (id, page url, photographer, sha256). Existing files are kept.
"""

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

KEYWORDS = ("beachwear", "summer outfit", "sundress", "seaside fashion")
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def select_ids(keyword_rows, keywords):
    wanted = {k.lower() for k in keywords}
    return sorted({row["photo_id"] for row in keyword_rows if row["keyword"].strip().lower() in wanted})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--keyword", action="append")
    parser.add_argument("--size", type=int, default=1920)
    args = parser.parse_args(argv)
    index, out = Path(args.index), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = set(select_ids(read_tsv(index / "keywords.tsv000"), args.keyword or KEYWORDS))
    manifest = []
    for row in read_tsv(index / "photos.tsv000"):
        if row["photo_id"] not in ids:
            continue
        target = out / f"{row['photo_id']}.jpg"
        if not target.exists():
            url = f"{row['photo_image_url']}?fm=jpg&q=95&fit=max&w={args.size}&h={args.size}"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sr-lab"}), timeout=60) as response:
                    data = response.read()
            except OSError as error:
                print("skip", row["photo_id"], error, flush=True)
                continue
            target.write_bytes(data)
        manifest.append({"id": row["photo_id"], "page": row["photo_url"], "photographer": row["photographer_username"],
                         "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        print(len(manifest), row["photo_id"], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
