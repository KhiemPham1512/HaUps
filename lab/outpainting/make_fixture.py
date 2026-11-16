"""Build a frozen anime evaluation fixture directory from a list of source image paths.

CLI:
  python .local-artifacts/outpainting/make_fixture.py --dataset-id ID --out DIRNAME --sources LISTFILE --note TEXT
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps

LAB = Path(__file__).resolve().parents[2] / ".local-artifacts" / "outpainting"
MIN_SOURCES = 1
MAX_SOURCES = 16


def _validate_out_name(lab_path: Path, out_name: str) -> Path:
    if (
        not isinstance(out_name, str)
        or not out_name.strip()
        or out_name != out_name.strip()
        or "/" in out_name
        or "\\" in out_name
        or ".." in out_name
        or out_name in (".", "..")
    ):
        raise ValueError(f"Invalid output directory name: {out_name!r}")
    rel = Path(out_name)
    if rel.is_absolute() or len(rel.parts) != 1 or rel.parts[0] in (".", ".."):
        raise ValueError(f"Invalid output directory name: {out_name!r}")
    out_dir = lab_path / out_name
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out_name}")
    resolved_lab = lab_path.resolve()
    resolved_out = out_dir.resolve()
    if (
        not resolved_out.is_relative_to(resolved_lab)
        or resolved_out.parent != resolved_lab
    ):
        raise ValueError(f"Output directory escapes lab root: {out_name!r}")
    return out_dir


def _validate_source_path(raw_source, index: int) -> Path:
    if not isinstance(raw_source, (str, Path)):
        raise ValueError(f"Source #{index} is not a valid path")
    raw_str = str(raw_source).strip()
    if not raw_str:
        raise ValueError(f"Source #{index} path is empty")
    src_path = Path(raw_str)
    if not src_path.is_absolute():
        raise ValueError(f"Source #{index} must be an absolute path")
    if any(part.lower() == "remove" for part in src_path.parts):
        raise ValueError(f"Source #{index} is inside an excluded 'remove' directory")
    if src_path.is_symlink():
        raise ValueError(f"Source #{index} must not be a symlink")
    resolved = src_path.resolve()
    if any(part.lower() == "remove" for part in resolved.parts):
        raise ValueError(f"Source #{index} resolves inside an excluded 'remove' directory")
    if not resolved.is_file() or not src_path.is_file():
        raise FileNotFoundError(f"Source #{index} is not a regular file")
    return resolved


def read_sources_list(list_file) -> list[Path]:
    """Read UTF-8 source list file (one absolute path per non-empty line)."""
    list_path = Path(list_file)
    if list_path.is_symlink() or not list_path.is_file():
        raise FileNotFoundError("Sources list file not found")
    lines = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return [Path(line) for line in lines]


def build_fixture(lab, dataset_id: str, out_name: str, sources, note: str) -> Path:
    """Create a frozen evaluation fixture under lab/out_name without leaking source paths."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("note must be a non-empty string")
    if not isinstance(sources, (list, tuple)):
        sources = list(sources)
    if not (MIN_SOURCES <= len(sources) <= MAX_SOURCES):
        raise ValueError(
            f"Number of sources must be between {MIN_SOURCES} and {MAX_SOURCES}, got {len(sources)}"
        )

    lab_path = Path(lab)
    if not lab_path.is_dir():
        raise FileNotFoundError("Lab directory does not exist")
    out_dir = _validate_out_name(lab_path, out_name)
    validated_sources = [
        _validate_source_path(src, idx) for idx, src in enumerate(sources, start=1)
    ]

    out_dir.mkdir(parents=False, exist_ok=False)
    try:
        files_records = []
        for idx, src_path in enumerate(validated_sources, start=1):
            raw_bytes = src_path.read_bytes()
            original_file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

            with Image.open(io.BytesIO(raw_bytes)) as raw_image:
                oriented_rgb = ImageOps.exif_transpose(raw_image).convert("RGB")

            width, height = oriented_rgb.size
            source_rgb_sha256 = hashlib.sha256(oriented_rgb.tobytes()).hexdigest()

            group = f"eval-{idx}"
            file_name = f"{group}.png"
            dest_path = out_dir / file_name
            oriented_rgb.save(dest_path, format="PNG")
            png_sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()

            files_records.append(
                {
                    "group": group,
                    "file": file_name,
                    "width": width,
                    "height": height,
                    "sha256": png_sha256,
                    "source_rgb_sha256": source_rgb_sha256,
                    "original_file_sha256": original_file_sha256,
                }
            )

        selected_at_utc = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        manifest = {
            "dataset": dataset_id,
            "frozen": True,
            "selected_at_utc": selected_at_utc,
            "purpose": "evaluation_only_not_training",
            "blind_holdout": False,
            "selection_note": note,
            "files": files_records,
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return out_dir
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier")
    parser.add_argument(
        "--out", required=True, help="Single directory name under lab root"
    )
    parser.add_argument(
        "--sources",
        required=True,
        help="UTF-8 file with one absolute image path per line",
    )
    parser.add_argument(
        "--note", required=True, help="Selection note recorded in manifest.json"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sources = read_sources_list(args.sources)
    build_fixture(
        LAB,
        dataset_id=args.dataset_id,
        out_name=args.out,
        sources=sources,
        note=args.note,
    )
    print(f"Built fixture {args.out} ({len(sources)} images)", flush=True)


if __name__ == "__main__":
    main()
