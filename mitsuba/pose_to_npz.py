#!/usr/bin/env python3

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Blender meta.json camera data into Mitsuba camview files."
    )
    parser.add_argument("filepath", type=Path, help="Path to the meta.json file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write camviewXXXX.npz files into",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


def read_json_file(filepath: Path) -> dict:
    with filepath.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)

    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a valid PNG file")

    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def infer_resolution(meta_path: Path, data: dict, width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("Both --width and --height must be provided together")
        return width, height

    meta_width = data.get("w")
    meta_height = data.get("h")
    if meta_width is not None and meta_height is not None:
        return int(meta_width), int(meta_height)

    prefix = meta_path.name.split(".", 1)[0]
    candidates = sorted(meta_path.parent.glob(f"{prefix}.*.rgb.png"))
    if not candidates:
        candidates = sorted(meta_path.parent.glob("*.rgb.png"))

    if not candidates:
        raise ValueError(
            f"Could not infer image resolution for {meta_path}. "
            "Add w/h to the meta.json or pass --width/--height."
        )

    return read_png_size(candidates[0])


def intrinsics_from_fovx(fovx: float, width: int, height: int) -> np.ndarray:
    fx = 0.5 * float(width) / math.tan(0.5 * float(fovx))
    fy = fx
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def main():
    args = parse_args()
    meta_path = args.filepath.expanduser().resolve()
    data = read_json_file(meta_path)
    frames = data["frames"]
    width, height = infer_resolution(meta_path, data, args.width, args.height)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (meta_path.parent / "frames" / "camview" / "camera_0").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving Mitsuba camviews to: {output_dir}")
    print(f"Image resolution: {width}x{height}")

    hw = np.array([height, width], dtype=np.int32)
    for index, frame in enumerate(frames):
        fovx = frame.get("fov", data["camera_angle_x"])
        camview_path = output_dir / f"camview{index:04d}.npz"
        np.savez(
            camview_path,
            K=intrinsics_from_fovx(fovx, width, height),
            T=np.array(frame["transform_matrix"], dtype=np.float32),
            HW=hw,
        )


if __name__ == "__main__":
    main()
