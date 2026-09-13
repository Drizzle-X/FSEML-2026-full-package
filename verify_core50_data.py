from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

from PIL import Image


EXPECTED_IMAGES = 164_866
EXPECTED_PATHS_SHA256 = "f4c03f616b52d073f3253d719b10d65a928d009c1feb3cc90603d9189a81d4b8"
METADATA_SHA256 = {
    "paths.pkl": "073e9a900fa558a31e9b9264b0cad7a3339c7b6d14ced0f602200ada1d2b694f",
    "LUP.pkl": "6aca5cd9af1ee9850bd0e6f25e865be2719a888faa2f708dd1142af16e5fa545",
    "labels.pkl": "123dcd893c3e4187d51712c1e7c1c809e4ddb06d56eb3aa4aaf42a7ad4f5c46c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    nested = root / "core50_128x128"
    if nested.is_dir() and not (root / "s1").is_dir():
        return nested
    return root


def verify_metadata(metadata_root: Path) -> list[str]:
    for name, expected in METADATA_SHA256.items():
        path = metadata_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing official metadata: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")

    with (metadata_root / "paths.pkl").open("rb") as handle:
        paths = pickle.load(handle)
    if len(paths) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} official paths, found {len(paths)}")
    digest = hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
    if digest != EXPECTED_PATHS_SHA256:
        raise ValueError(
            f"Official path-list SHA-256 mismatch: expected {EXPECTED_PATHS_SHA256}, got {digest}"
        )
    return paths


def image_indices(paths: list[str], mode: str) -> list[int]:
    if mode == "none":
        return []
    if mode == "all":
        return list(range(len(paths)))
    selected = []
    seen = set()
    for index, relative in enumerate(paths):
        key = tuple(Path(relative).parts[:2])
        if key not in seen:
            seen.add(key)
            selected.append(index)
    return selected


def verify_images(image_root: Path, paths: list[str], mode: str) -> None:
    missing = [relative for relative in paths if not (image_root / relative).is_file()]
    if missing:
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(paths)} official RGB images. First missing paths:\n  {preview}"
        )

    indices = image_indices(paths, mode)
    for index in indices:
        path = image_root / paths[index]
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (128, 128) or image.mode != "RGB":
                raise ValueError(f"Unexpected image format for {path}: size={image.size}, mode={image.mode}")
    print(f"Image files present: {len(paths)}; decoded and checked: {len(indices)} ({mode})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an official CORe50 128x128 extraction.")
    parser.add_argument("--data-root", required=True, help="Extracted core50_128x128 directory or its parent.")
    parser.add_argument("--metadata-root", default="vendor/core50_official")
    parser.add_argument(
        "--image-check",
        choices=("none", "sample", "all"),
        default="sample",
        help="Decode no images, one image per session/object directory, or every image.",
    )
    parser.add_argument("--archive", help="Optional downloaded ZIP; report its SHA-256 for provenance.")
    args = parser.parse_args()

    metadata_root = Path(args.metadata_root).expanduser().resolve()
    image_root = resolve_image_root(Path(args.data_root))
    paths = verify_metadata(metadata_root)
    print(f"Metadata SHA-256 checks passed: {len(METADATA_SHA256)} files")
    print(f"Official path-list SHA-256: {EXPECTED_PATHS_SHA256}")
    verify_images(image_root, paths, args.image_check)
    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        print(f"Downloaded archive SHA-256 (record for provenance): {sha256(archive)}  {archive}")
    print(f"CORe50 verification passed: {image_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"CORe50 verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
