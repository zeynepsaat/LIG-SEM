from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def find_images(source: Path) -> Sequence[Path]:
    if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [source]

    if source.is_dir():
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise FileNotFoundError(f"No supported images found at: {source}")


def crop_image(path: Path, output_dir: Path, left: int, top: int, width: int, height: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(path) as img:
        right = left + width
        bottom = top + height

        # Clamp crop box to image bounds to avoid PIL errors when values overflow.
        left_clamped = max(0, min(left, img.width))
        top_clamped = max(0, min(top, img.height))
        right_clamped = max(left_clamped, min(right, img.width))
        bottom_clamped = max(top_clamped, min(bottom, img.height))

        cropped = img.crop((left_clamped, top_clamped, right_clamped, bottom_clamped))

        output_path = output_dir / f"{path.stem}_crop{path.suffix}"
        cropped.save(output_path)
        return output_path


def process_images(
    paths: Iterable[Path],
    output_dir: Path,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    source_root: Path | None = None,
) -> None:
    for img_path in paths:
        target_dir = output_dir
        if source_root is not None:
            relative_parent = img_path.parent.relative_to(source_root)
            target_dir = output_dir / relative_parent

        result = crop_image(img_path, target_dir, left, top, width, height)
        print(f"Saved: {result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop TIFF or other images with custom dimensions.")
    parser.add_argument("source", type=Path, help="Image file or directory containing images.")
    parser.add_argument("output", type=Path, help="Directory to write cropped images.")
    parser.add_argument("left", type=int, help="Left offset (pixels) from the top-left corner.")
    parser.add_argument("top", type=int, help="Top offset (pixels) from the top-left corner.")
    parser.add_argument("width", type=int, help="Crop width in pixels.")
    parser.add_argument("height", type=int, help="Crop height in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = find_images(args.source)
    if not images:
        raise SystemExit("No images matched the given source and supported extensions.")

    source_root = args.source if args.source.is_dir() else None
    process_images(
        images,
        args.output,
        args.left,
        args.top,
        args.width,
        args.height,
        source_root=source_root,
    )


if __name__ == "__main__":
    main()
