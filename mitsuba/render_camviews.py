#!/usr/bin/env python3

import argparse
import math
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_xml", type=Path)
    parser.add_argument("camview_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--variant", default="auto")
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--near-clip", type=float, default=0.1)
    parser.add_argument("--far-clip", type=float, default=10000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_variant(mi, requested: str) -> str:
    available = set(mi.variants())
    if requested != "auto":
        if requested not in available:
            raise ValueError(f"Requested Mitsuba variant {requested!r} is unavailable")
        return requested

    for candidate in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"):
        if candidate in available:
            return candidate
    raise RuntimeError(f"No usable RGB Mitsuba variant found in {sorted(available)}")


def mitsuba_camera_rotation() -> np.ndarray:
    rotation = np.eye(4, dtype=np.float64)
    rotation[0, 0] = -1.0
    rotation[2, 2] = -1.0
    return rotation


def principal_point_offsets(K: np.ndarray, width: int, height: int) -> tuple[float, float]:
    scale = float(max(width, height))
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    offset_x = (cx - width * 0.5) / scale
    offset_y = -(cy - height * 0.5) / scale
    return offset_x, offset_y


def sensor_dict(mi, camview: dict, spp: int, near_clip: float, far_clip: float) -> dict:
    K = camview["K"]
    T_blender = camview["T"]
    height, width = [int(x) for x in camview["HW"]]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    if width >= height:
        fov_axis = "x"
        fov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    else:
        fov_axis = "y"
        fov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    to_world = T_blender @ mitsuba_camera_rotation()
    offset_x, offset_y = principal_point_offsets(K, width, height)

    return {
        "type": "perspective",
        "fov_axis": fov_axis,
        "fov": fov,
        "principal_point_offset_x": offset_x,
        "principal_point_offset_y": offset_y,
        "near_clip": near_clip,
        "far_clip": far_clip,
        "to_world": mi.ScalarTransform4f(to_world),
        "sampler": {
            "type": "independent",
            "sample_count": spp,
        },
        "film": {
            "type": "hdrfilm",
            "width": width,
            "height": height,
            "pixel_format": "rgb",
            "rfilter": {"type": "box"},
        },
    }


def load_camview(camview_path: Path) -> dict:
    payload = np.load(camview_path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        return {key: payload[key] for key in payload.files}

    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        item = payload.item()
        if isinstance(item, dict):
            return item

    raise TypeError(
        f"Unsupported camview file format for {camview_path}. "
        "Expected .npz with K/T/HW arrays."
    )


def main() -> int:
    args = parse_args()

    scene_xml = args.scene_xml.expanduser().resolve()
    camview_dir = args.camview_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not scene_xml.exists():
        raise FileNotFoundError(scene_xml)
    if not camview_dir.exists():
        raise FileNotFoundError(camview_dir)

    if output_dir.exists():
        if not args.overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {output_dir} is not empty. Pass --overwrite to reuse it."
            )
    else:
        output_dir.mkdir(parents=True)

    import mitsuba as mi

    variant = choose_variant(mi, args.variant)
    mi.set_variant(variant)
    scene = mi.load_file(str(scene_xml))

    camviews = sorted(camview_dir.glob("*.npz"))
    if not camviews:
        camviews = sorted(camview_dir.glob("*.npy"))
    if not camviews:
        raise FileNotFoundError(f"No camview .npz or .npy files found in {camview_dir}")

    print(f"scene_xml: {scene_xml}")
    print(f"variant: {variant}")
    print(f"spp: {args.spp}")
    print(f"camviews: {len(camviews)}")
    print(f"output_dir: {output_dir}")

    for index, camview_path in enumerate(camviews):
        camview = load_camview(camview_path)
        sensor = mi.load_dict(
            sensor_dict(
                mi,
                camview,
                spp=args.spp,
                near_clip=args.near_clip,
                far_clip=args.far_clip,
            )
        )

        image = mi.render(scene, sensor=sensor, spp=args.spp, seed=args.seed + index)

        stem = camview_path.stem
        exr_path = output_dir / f"{stem}.exr"
        png_path = output_dir / f"{stem}.png"
        mi.util.write_bitmap(str(exr_path), image, write_async=False)
        mi.util.write_bitmap(str(png_path), image, write_async=False)

        reference_png = (
            camview_path.parents[2]
            / "Image"
            / camview_path.parent.name
            / f"Image{stem[len('camview'):]}.png"
        )
        if reference_png.exists():
            shutil.copy2(reference_png, output_dir / f"{stem}_reference.png")

        print(f"rendered: {camview_path.name} -> {png_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
