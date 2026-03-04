#!/usr/bin/env python3
"""
Scan a folder or asset list and record how expensive each asset is to export to Mitsuba.

Run under Blender, for example:
  blender -b --python mitsuba/scan_asset_export_cost.py -- assets/glbs asset_export_cost.json
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import bpy


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
EXPORT_SCRIPT = REPO_ROOT / "mitsuba" / "export_mitsuba.py"
INFINIGEN_ROOT = Path("/home/vhchu/infinigen")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INFINIGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(INFINIGEN_ROOT))

from utils import blender_utils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def load_export_module():
    spec = importlib.util.spec_from_file_location("export_mitsuba_mod", EXPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_asset_paths(input_path: Path):
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        paths = []
        for pattern in ("*.glb", "*.gltf", "*.obj"):
            paths.extend(input_path.glob(pattern))
        return sorted(path.resolve() for path in paths)

    if input_path.is_file():
        with input_path.open("r", encoding="utf-8") as handle:
            return [
                Path(line.strip()).expanduser().resolve()
                for line in handle
                if line.strip()
            ]

    raise FileNotFoundError(f"Could not resolve asset input path: {input_path}")


def classify_material_reason(material, export_mod, materials_module, export_context_module):
    surface_node = export_mod.get_material_surface_node(material)
    if surface_node is None:
        return None

    probe_ctx = export_mod.make_material_probe_context(export_context_module)
    try:
        materials_module.cycles_material_to_dict(probe_ctx, surface_node)
        return None
    except Exception as err:
        return f"{type(err).__name__}: {err}"


def analyze_asset(asset_path, export_mod, materials_module, export_context_module, infinigen_export):
    blender_utils.clear_scene()
    asset = None
    started_at = time.perf_counter()

    try:
        asset = blender_utils.add_object_file(
            str(asset_path), with_empty=True, recenter=True, rescale=True
        )
        mesh_objects = [obj for obj in asset.objs if obj.type == "MESH" and not obj.hide_render]

        material_needs_bake_cache = {}
        material_reason_cache = {}
        baked_material_names = set()
        baked_reason_counts = Counter()
        baked_material_usage = Counter()

        skipped_meshes = 0
        clean_meshes = 0
        baked_meshes = 0

        for obj in mesh_objects:
            if infinigen_export.skipBake(obj):
                skipped_meshes += 1
                continue

            object_needs_bake = False
            seen_materials = set()
            for slot in obj.material_slots:
                material = slot.material
                if material is None:
                    continue
                key = material.as_pointer()
                if key not in material_needs_bake_cache:
                    reason = classify_material_reason(
                        material, export_mod, materials_module, export_context_module
                    )
                    material_reason_cache[key] = reason
                    material_needs_bake_cache[key] = reason is not None
                if not material_needs_bake_cache[key] or key in seen_materials:
                    continue

                seen_materials.add(key)
                object_needs_bake = True
                baked_material_names.add(material.name)
                baked_material_usage[material.name] += 1
                baked_reason_counts[material_reason_cache[key]] += 1

            if object_needs_bake:
                baked_meshes += 1
            else:
                clean_meshes += 1

        analyzable_meshes = clean_meshes + baked_meshes
        baked_ratio = baked_meshes / analyzable_meshes if analyzable_meshes else 0.0
        score = baked_meshes + 25 * len(baked_material_names)

        return {
            "asset_path": str(asset_path),
            "status": "ok",
            "mesh_objects": len(mesh_objects),
            "skipped_mesh_objects": skipped_meshes,
            "clean_mesh_objects": clean_meshes,
            "baked_mesh_objects": baked_meshes,
            "baked_object_ratio": baked_ratio,
            "unique_materials": int(asset.get_num_materials()),
            "unique_baked_materials": len(baked_material_names),
            "export_cost_score": score,
            "top_baked_materials": [
                {"name": name, "object_count": count}
                for name, count in baked_material_usage.most_common(20)
            ],
            "baked_material_reasons": dict(sorted(baked_reason_counts.items())),
            "analysis_seconds": time.perf_counter() - started_at,
        }
    except Exception as err:
        return {
            "asset_path": str(asset_path),
            "status": "error",
            "error": f"{type(err).__name__}: {err}",
            "analysis_seconds": time.perf_counter() - started_at,
        }
    finally:
        try:
            if asset is not None:
                asset.clear_objects()
        except Exception:
            pass
        blender_utils.clear_scene()
        try:
            bpy.ops.outliner.orphans_purge(do_recursive=True)
        except Exception:
            pass


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path).expanduser().resolve()

    export_mod = load_export_module()
    export_mod.ensure_mitsuba_importable()
    export_mod.ensure_infinigen_export_importable()

    asset_paths = resolve_asset_paths(input_path)
    if args.limit is not None:
        asset_paths = asset_paths[: args.limit]

    addon_root = export_mod.prepare_addon_module(export_mod.ZIP_PATH)
    try:
        materials_module = importlib.import_module(
            f"{export_mod.ADDON_MODULE}.io.exporter.materials"
        )
        export_context_module = importlib.import_module(
            f"{export_mod.ADDON_MODULE}.io.exporter.export_context"
        )
        from infinigen.tools import export as infinigen_export

        entries = []
        started_at = time.perf_counter()
        total_assets = len(asset_paths)
        for index, asset_path in enumerate(asset_paths, start=1):
            print(f"[scan] {index}/{total_assets} {asset_path}")
            entry = analyze_asset(
                asset_path,
                export_mod,
                materials_module,
                export_context_module,
                infinigen_export,
            )
            entries.append(entry)
            if entry["status"] == "ok":
                print(
                    "[scan] "
                    f"mesh={entry['mesh_objects']} "
                    f"baked={entry['baked_mesh_objects']} "
                    f"unique_baked_materials={entry['unique_baked_materials']} "
                    f"score={entry['export_cost_score']}"
                )
            else:
                print(f"[scan][warn] {entry['error']}")

        payload = {
            "version": 1,
            "input_path": str(input_path.expanduser().resolve()),
            "asset_count": len(entries),
            "generated_seconds": time.perf_counter() - started_at,
            "entries": entries,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[scan] wrote manifest: {output_path}")
    finally:
        import shutil

        shutil.rmtree(addon_root, ignore_errors=True)


if __name__ == "__main__":
    main()
