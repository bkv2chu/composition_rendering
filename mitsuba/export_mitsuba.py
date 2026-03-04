import argparse
import importlib
import logging
import os
import pwd
import re
import shutil
import sys
import tempfile
import time
import types
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import bpy


ZIP_PATH = Path("/home/vhchu/infinigen/mitsuba-blender.zip")
ADDON_MODULE = "mitsuba_blender"
REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_TAGS = {
    "Sensor": "sensor",
    "Film": "film",
    "Emitter": "emitter",
    "Sampler": "sampler",
    "Shape": "shape",
    "Texture": "texture",
    "Volume": "volume",
    "Medium": "medium",
    "BSDF": "bsdf",
    "Integrator": "integrator",
    "PhaseFunction": "phase",
    "ReconstructionFilter": "rfilter",
}
POINT_KEYS = {"position", "center", "target", "origin"}
VECTOR_KEYS = {"direction", "up"}
MATERIAL_WARNING_RE = re.compile(
    r"Export of material '.*?' failed: (.*)\. Exporting a dummy material instead\."
)
UNSUPPORTED_OBJECT_RE = re.compile(r"Object: .* of type '([^']+)' is not supported!")
EMPTY_MESH_RE = re.compile(r"Mesh: .* has no faces\. Skipping\.")
SOFT_LIGHT_RE = re.compile(
    r"Light '.*' has a non-zero soft shadow radius\. It will be ignored\."
)
WORLD_WARNING_RE = re.compile(r"Error while exporting world: (.*)\. Not exporting it\.")
CYCLES_GPU_TYPES_PREFERENCE = ["OPTIX", "CUDA", "METAL", "HIP", "ONEAPI", "CPU"]


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("export_path", nargs="?", default="exported_scene.xml")
    parser.add_argument("--skip-bake-materials", action="store_true")
    parser.add_argument("--bake-all-materials", action="store_true")
    parser.add_argument("--bake-normal-maps", action="store_true")
    parser.add_argument("--bake-special-maps", action="store_true")
    parser.add_argument("--bake-resolution", type=int, default=512)
    parser.add_argument("--reference-xml")
    return parser.parse_args(argv)


def prepare_addon_module(zip_path: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Could not find Mitsuba add-on archive at {zip_path}")

    extract_root = Path(tempfile.mkdtemp(prefix="mitsuba_blender_"))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_root)

    addon_source = None
    for init_file in extract_root.rglob("__init__.py"):
        parent = init_file.parent
        if (parent / "io").is_dir() and (parent / "engine").is_dir():
            addon_source = parent
            break

    if addon_source is None:
        raise RuntimeError("Could not locate Mitsuba Blender add-on package in archive")

    target_dir = extract_root / ADDON_MODULE
    if addon_source != target_dir:
        shutil.move(str(addon_source), str(target_dir))

    if str(extract_root) not in sys.path:
        sys.path.insert(0, str(extract_root))

    purge_addon_modules()
    patch_addon_source(extract_root / ADDON_MODULE)

    return extract_root


def purge_addon_modules():
    for module_name in list(sys.modules):
        if module_name == ADDON_MODULE or module_name.startswith(f"{ADDON_MODULE}."):
            del sys.modules[module_name]


def patch_addon_source(addon_dir: Path):
    geometry_path = addon_dir / "io" / "exporter" / "geometry.py"
    geometry_source = geometry_path.read_text()
    if "import struct\n" not in geometry_source:
        geometry_source = geometry_source.replace("import bpy\n", "import bpy\nimport struct\n")

    fallback_helper = """\n\nclass DirectPlyMesh:\n    def __init__(self, vertices, faces, has_uvs):\n        self.vertices = vertices\n        self.faces = faces\n        self.has_uvs = has_uvs\n\n    def face_count(self):\n        return len(self.faces)\n\n    def has_vertex_normals(self):\n        return False\n\n    def write_ply(self, filepath):\n        header = [\n            'ply',\n            'format binary_little_endian 1.0',\n            f'element vertex {len(self.vertices)}',\n            'property float x',\n            'property float y',\n            'property float z',\n        ]\n        if self.has_uvs:\n            header.extend([\n                'property float s',\n                'property float t',\n            ])\n        header.extend([\n            f'element face {len(self.faces)}',\n            'property list uchar int vertex_indices',\n            'end_header',\n            '',\n        ])\n\n        with open(filepath, 'wb') as handle:\n            handle.write('\\n'.join(header).encode('ascii'))\n            if self.has_uvs:\n                for x, y, z, s, t in self.vertices:\n                    handle.write(struct.pack('<5f', x, y, z, s, t))\n            else:\n                for x, y, z in self.vertices:\n                    handle.write(struct.pack('<3f', x, y, z))\n\n            for face in self.faces:\n                handle.write(struct.pack('<B3i', 3, *face))\n\n\ndef build_direct_ply_mesh(export_ctx, b_mesh, matrix_world, name, mat_nr):\n    uv_layer = None\n    for layer in b_mesh.uv_layers:\n        if layer.active_render:\n            uv_layer = layer\n            break\n\n    vertices = []\n    faces = []\n    for tri in b_mesh.loop_triangles:\n        poly = b_mesh.polygons[tri.polygon_index]\n        if mat_nr >= 0 and poly.material_index != mat_nr:\n            continue\n\n        face = []\n        for loop_index, vertex_index in zip(tri.loops, tri.vertices):\n            coord = b_mesh.vertices[vertex_index].co.copy()\n            if matrix_world:\n                coord = matrix_world @ coord\n\n            if uv_layer is not None:\n                uv = uv_layer.data[loop_index].uv\n                vertices.append((coord.x, coord.y, coord.z, uv.x, uv.y))\n            else:\n                vertices.append((coord.x, coord.y, coord.z))\n            face.append(len(vertices) - 1)\n        faces.append(tuple(face))\n\n    if not faces:\n        export_ctx.log(f\"Mesh: {name} has no faces. Skipping.\", 'WARN')\n        return None\n\n    return DirectPlyMesh(vertices, faces, uv_layer is not None)\n"""
    if "class DirectPlyMesh:" not in geometry_source:
        geometry_source = geometry_source.replace(
            "\n\ndef convert_mesh(export_ctx, b_mesh, matrix_world, name, mat_nr):",
            fallback_helper + "\n\ndef convert_mesh(export_ctx, b_mesh, matrix_world, name, mat_nr):",
        )

    broken_block = """        if mat_count == 0: # No assigned material\n            converted_parts.append((\n                name_clean,\n                -1,\n                convert_mesh(export_ctx, b_mesh, transform, name_clean, 0)\n            ))\n"""
    fixed_block = """        if mat_count == 0: # No assigned material\n            mts_mesh = convert_mesh(export_ctx, b_mesh, transform, name_clean, 0)\n            if mts_mesh is not None:\n                converted_parts.append((\n                    name_clean,\n                    -1,\n                    mts_mesh\n                ))\n"""
    if broken_block in geometry_source:
        geometry_source = geometry_source.replace(broken_block, fixed_block)

    invalid_normals_block = """    # Return the mitsuba mesh\n    return load_dict(props)\n"""
    invalid_normals_fix = """    # Return the mitsuba mesh. Some Blender meshes carry invalid vertex\n    # normal buffers that make Mitsuba's BlenderMesh loader abort.\n    # Fall back to a direct binary PLY writer for those meshes.\n    try:\n        return load_dict(props)\n    except RuntimeError as exc:\n        if 'invalid normals' not in str(exc).lower():\n            raise\n        export_ctx.log(\n            f\"Mesh: {name} has invalid vertex normals. Falling back to direct PLY export.\",\n            'WARN'\n        )\n        return build_direct_ply_mesh(export_ctx, b_mesh, matrix_world, name, mat_nr)\n"""
    if invalid_normals_block in geometry_source:
        geometry_source = geometry_source.replace(
            invalid_normals_block, invalid_normals_fix
        )

    geometry_path.write_text(geometry_source)

    init_path = addon_dir / "io" / "exporter" / "__init__.py"
    init_source = init_path.read_text()
    if "import types\n" not in init_source:
        init_source = init_source.replace("import os\n", "import os\nimport types\n")

    light_block = """        progress_counter = 0\n        # Main export loop\n        for object_instance in depsgraph.object_instances:\n"""
    patched_light_block = """        progress_counter = 0\n        seen_light_names = set()\n        # Main export loop\n        for object_instance in depsgraph.object_instances:\n"""
    if light_block in init_source:
        init_source = init_source.replace(light_block, patched_light_block)

    export_light_block = """            elif object_type == 'LIGHT':\n                lights.export_light(object_instance, self.export_ctx)\n"""
    patched_export_light_block = """            elif object_type == 'LIGHT':\n                lights.export_light(object_instance, self.export_ctx)\n                seen_light_names.add(evaluated_obj.name_full)\n"""
    if export_light_block in init_source:
        init_source = init_source.replace(export_light_block, patched_export_light_block)

    tail_block = """            else:\n                self.export_ctx.log(\"Object: %s of type '%s' is not supported!\" % (evaluated_obj.name_full, object_type), 'WARN')\n"""
    patched_tail_block = """            else:\n                self.export_ctx.log(\"Object: %s of type '%s' is not supported!\" % (evaluated_obj.name_full, object_type), 'WARN')\n\n        for light_obj in b_scene.objects:\n            if light_obj.type != 'LIGHT' or light_obj.name_full in seen_light_names:\n                continue\n            if light_obj.hide_render:\n                self.export_ctx.log(\"Object: {} is hidden for render. Ignoring it.\".format(light_obj.name), 'INFO')\n                continue\n            if self.use_selection and not light_obj.original.select_get():\n                continue\n            lights.export_light(types.SimpleNamespace(object=light_obj.evaluated_get(depsgraph)), self.export_ctx)\n"""
    if tail_block in init_source:
        init_source = init_source.replace(tail_block, patched_tail_block)

    init_path.write_text(init_source)


def find_mitsuba_parent() -> Path:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    actual_home = Path(pwd.getpwuid(os.getuid()).pw_dir)

    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if base := os.environ.get(env_var):
            candidates.append(Path(base) / "lib" / pyver / "site-packages")

    candidates.append(Path.home() / ".local" / "lib" / pyver / "site-packages")
    candidates.append(actual_home / ".local" / "lib" / pyver / "site-packages")
    candidates.extend(Path(p) for p in sys.path if "site-packages" in p)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "mitsuba").exists():
            return candidate

    raise ModuleNotFoundError(
        "Could not locate the Mitsuba Python package. "
        "Set up a venv/conda env with Mitsuba or install it under ~/.local."
    )


def ensure_mitsuba_importable():
    try:
        import mitsuba as mi

        return mi
    except ModuleNotFoundError:
        parent = find_mitsuba_parent()
        sys.path.insert(0, str(parent))
        import mitsuba as mi

        return mi


def ensure_infinigen_export_importable():
    if "gin" not in sys.modules:
        gin_module = types.ModuleType("gin")
        gin_module.configurable = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
        sys.modules["gin"] = gin_module

    if "trimesh" not in sys.modules:
        sys.modules["trimesh"] = types.ModuleType("trimesh")

    if "infinigen.core.util.blender" not in sys.modules:
        blender_module = types.ModuleType("infinigen.core.util.blender")

        class SelectObjects:
            def __init__(self, *objects, active=None):
                self.objects = list(objects)
                self.active_index = active
                self.previous_active = None
                self.previous_selection = []

            def __enter__(self):
                self.previous_active = bpy.context.view_layer.objects.active
                self.previous_selection = list(bpy.context.selected_objects)
                bpy.ops.object.select_all(action="DESELECT")
                for obj in self.objects:
                    obj.select_set(True)
                if self.objects:
                    index = 0 if self.active_index is None else self.active_index
                    bpy.context.view_layer.objects.active = self.objects[index]
                return self

            def __exit__(self, exc_type, exc, tb):
                bpy.ops.object.select_all(action="DESELECT")
                for obj in self.previous_selection:
                    if obj.name in bpy.data.objects:
                        obj.select_set(True)
                if self.previous_active and self.previous_active.name in bpy.data.objects:
                    bpy.context.view_layer.objects.active = self.previous_active
                return False

        blender_module.SelectObjects = SelectObjects

        util_module = types.ModuleType("infinigen.core.util")
        util_module.blender = blender_module
        sys.modules["infinigen.core.util"] = util_module
        sys.modules["infinigen.core.util.blender"] = blender_module


def configure_cycles_for_baking(image_res: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 1
        scene.cycles.tile_x = image_res
        scene.cycles.tile_y = image_res
        scene.cycles.use_denoising = False
        try:
            scene.cycles.use_preview_denoising = False
        except Exception:
            pass

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None or not hasattr(scene, "cycles"):
        print("[mitsuba-export][warn] Cycles preferences unavailable, baking on CPU")
        scene.cycles.device = "CPU"
        return []

    prefs = addon.preferences
    print(
        "[mitsuba-export] bake env: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}"
    )
    if hasattr(prefs, "get_device_types") and hasattr(prefs, "get_devices_for_type"):
        for device_type in prefs.get_device_types(bpy.context):
            prefs.get_devices_for_type(device_type[0])
    elif hasattr(prefs, "refresh_devices"):
        prefs.refresh_devices()

    devices = list(getattr(prefs, "devices", []))
    if not devices:
        print("[mitsuba-export][warn] No Cycles devices found, baking on CPU")
        scene.cycles.device = "CPU"
        return []

    print(
        "[mitsuba-export] cycles devices: "
        + ", ".join(
            f"{device.name}[type={device.type}, use={getattr(device, 'use', False)}]"
            for device in devices
        )
    )

    types = sorted(
        {device.type for device in devices},
        key=lambda item: (
            CYCLES_GPU_TYPES_PREFERENCE.index(item)
            if item in CYCLES_GPU_TYPES_PREFERENCE
            else len(CYCLES_GPU_TYPES_PREFERENCE)
        ),
    )
    use_device_type = next((device_type for device_type in types if device_type != "CPU"), "CPU")

    for device in devices:
        device.use = False

    if use_device_type == "CPU":
        scene.cycles.device = "CPU"
        cpu_devices = [device for device in devices if device.type == "CPU"]
        for device in cpu_devices:
            device.use = True
        print("[mitsuba-export][warn] Only CPU Cycles devices found")
        return cpu_devices

    prefs.compute_device_type = use_device_type
    if hasattr(prefs, "get_devices_for_type"):
        prefs.get_devices_for_type(use_device_type)
    scene.cycles.device = "GPU"
    use_devices = [device for device in devices if device.type == use_device_type]
    for device in use_devices:
        device.use = True

    print(
        "[mitsuba-export] cycles baking device: "
        f"{use_device_type} ({', '.join(device.name for device in use_devices)})"
    )
    return use_devices


def format_seconds(duration: float) -> str:
    return f"{duration:.2f}s"


def clear_export_uv_layers(obj):
    if obj.type != "MESH":
        return
    for uv_layer in list(obj.data.uv_layers):
        if uv_layer.name.startswith("ExportUV"):
            obj.data.uv_layers.remove(uv_layer)


def normalize_world_mapping_modes():
    world = bpy.context.scene.world
    if world is None or not world.use_nodes or world.node_tree is None:
        return []

    changed_nodes = []
    for node in world.node_tree.nodes:
        if node.bl_idname != "ShaderNodeMapping":
            continue
        if getattr(node, "vector_type", None) == "TEXTURE":
            continue
        if not any(
            link.to_node.bl_idname == "ShaderNodeTexEnvironment"
            for output in node.outputs
            for link in output.links
        ):
            continue

        changed_nodes.append((node, node.vector_type))
        print(
            "[mitsuba-export][warn] "
            f"world mapping node {node.name!r} uses unsupported mode {node.vector_type!r}; "
            "temporarily switching it to 'TEXTURE' for Mitsuba export."
        )
        node.vector_type = "TEXTURE"

    return changed_nodes


def restore_world_mapping_modes(changed_nodes):
    for node, vector_type in changed_nodes:
        try:
            node.vector_type = vector_type
        except Exception:
            pass


def get_material_surface_node(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None

    output = material.node_tree.nodes.get("Material Output")
    if output is None or not output.inputs["Surface"].is_linked:
        return None

    return output.inputs["Surface"].links[0].from_node


def make_material_probe_context(export_context_module):
    probe_ctx = export_context_module.ExportContext()
    probe_ctx.export_texture = types.MethodType(
        lambda self, image: f"textures/{image.name}", probe_ctx
    )
    probe_ctx.log = types.MethodType(lambda self, message, level="INFO": None, probe_ctx)
    return probe_ctx


def material_needs_bake(material, materials_module, export_context_module):
    surface_node = get_material_surface_node(material)
    if surface_node is None:
        return False

    probe_ctx = make_material_probe_context(export_context_module)
    try:
        materials_module.cycles_material_to_dict(probe_ctx, surface_node)
        return False
    except NotImplementedError:
        return True
    except Exception as err:
        print(
            "[mitsuba-export][warn] "
            f"material probe failed on {material.name}: {type(err).__name__}: {err}. "
            "Baking it as a fallback."
        )
        return True


def object_needs_bake(obj, materials_module, export_context_module, material_cache):
    for slot in obj.material_slots:
        material = slot.material
        if material is None:
            continue

        cache_key = material.as_pointer()
        if cache_key not in material_cache:
            material_cache[cache_key] = material_needs_bake(
                material, materials_module, export_context_module
            )

        if material_cache[cache_key]:
            return True

    return False


def bake_scene_materials(
    image_res: int,
    bake_root: Path,
    materials_module,
    export_context_module,
    bake_all_materials: bool,
    bake_normal_maps: bool,
    bake_special_maps: bool,
):
    ensure_infinigen_export_importable()
    from infinigen.tools import export as infinigen_export

    logging.getLogger().setLevel(logging.WARNING)
    configure_cycles_for_baking(image_res)
    bake_root.mkdir(parents=True, exist_ok=True)
    original_bake_types = dict(infinigen_export.BAKE_TYPES)
    original_special_bake = dict(infinigen_export.SPECIAL_BAKE)
    if not bake_normal_maps:
        infinigen_export.BAKE_TYPES = {
            key: value
            for key, value in original_bake_types.items()
            if key != "NORMAL"
        }
    if not bake_special_maps:
        infinigen_export.SPECIAL_BAKE = {}
    print(
        "[mitsuba-export] bake passes: "
        f"{', '.join(infinigen_export.BAKE_TYPES.keys())} + "
        f"{', '.join(infinigen_export.SPECIAL_BAKE.keys())}"
    )

    visible_objects = set(bpy.context.view_layer.objects)
    material_cache = {}
    bake_queue = []
    baked = 0
    skipped = 0
    clean = 0
    failed = 0
    bake_start = time.perf_counter()

    try:
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj not in visible_objects or obj.hide_render:
                continue
            if infinigen_export.skipBake(obj):
                skipped += 1
                continue
            if not bake_all_materials and not object_needs_bake(
                obj, materials_module, export_context_module, material_cache
            ):
                clean += 1
                continue

            bake_queue.append(obj)

        print(
            "[mitsuba-export] bake plan: "
            f"{len(bake_queue)} queued, {clean} already exportable, {skipped} skipped"
        )

        for index, obj in enumerate(bake_queue, start=1):
            obj_start = time.perf_counter()
            print(
                "[mitsuba-export] bake start: "
                f"{index}/{len(bake_queue)} {obj.name}"
            )

            clear_export_uv_layers(obj)
            obj.hide_viewport = False

            try:
                infinigen_export.bake_object(obj, bake_root, image_res, export_usd=False)
                baked += 1
                print(
                    "[mitsuba-export] bake done: "
                    f"{obj.name} in {format_seconds(time.perf_counter() - obj_start)}"
                )
            except Exception as err:
                failed += 1
                print(
                    "[mitsuba-export][warn] "
                    f"bake failed on {obj.name} after "
                    f"{format_seconds(time.perf_counter() - obj_start)}: "
                    f"{type(err).__name__}: {err}"
                )
    finally:
        infinigen_export.BAKE_TYPES = original_bake_types
        infinigen_export.SPECIAL_BAKE = original_special_bake

    print(
        "[mitsuba-export] bake summary: "
        f"{baked} baked, {clean} already exportable, {skipped} skipped, "
        f"{failed} failed at {image_res}px in {format_seconds(time.perf_counter() - bake_start)}"
    )


def format_number(value) -> str:
    return f"{float(value):.9g}"


def format_float_list(values) -> str:
    return " ".join(format_number(v) for v in values)


class WriteXMLCompat:
    def __init__(self, filename, subfolders, split_files=False):
        self.filename = Path(filename)
        self.subfolders = subfolders
        self.split_files = split_files
        self.mi = ensure_mitsuba_importable()
        self.plugin_manager = self.mi.PluginManager.instance()

    def process(self, scene_data):
        root = ET.Element("scene", version="3.0.0")
        for key, value in scene_data.items():
            if key == "type":
                continue
            self._append_plugin(root, key, value, container_type="scene")

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(self.filename, encoding="utf-8", xml_declaration=True)

    def _plugin_tag(self, plugin_type: str) -> str:
        object_type = self.plugin_manager.plugin_type(plugin_type)
        try:
            return PLUGIN_TAGS[object_type.name]
        except KeyError as err:
            raise ValueError(
                f"Unsupported Mitsuba plugin category {object_type} for type {plugin_type}"
            ) from err

    def _append_plugin(self, parent, key, value, container_type: str):
        if not isinstance(value, dict) or "type" not in value:
            raise TypeError(f"Expected plugin dictionary for key {key!r}, got {type(value)}")

        plugin_type = value["type"]
        if plugin_type == "ref":
            attrs = {"id": str(value["id"])}
            if container_type not in {"scene", "shapegroup"} and key is not None:
                attrs["name"] = key
            ET.SubElement(parent, "ref", attrs)
            return

        tag = self._plugin_tag(plugin_type)
        attrs = {"type": plugin_type}

        if container_type in {"scene", "shapegroup"}:
            if key and not str(key).startswith("elm__"):
                attrs["id"] = str(key)
        elif key and key != tag:
            attrs["name"] = str(key)

        elem = ET.SubElement(parent, tag, attrs)
        for child_key, child_value in value.items():
            if child_key == "type":
                continue
            self._append_value(elem, child_key, child_value, parent_type=plugin_type)

    def _append_value(self, parent, key, value, parent_type: str):
        if isinstance(value, dict):
            value_type = value.get("type")
            if value_type in {"rgb", "spectrum"}:
                self._append_spectrum(parent, key, value)
            else:
                self._append_plugin(parent, key, value, container_type=parent_type)
            return

        if hasattr(value, "matrix"):
            transform = ET.SubElement(parent, "transform", {"name": key})
            matrix = value.matrix.numpy().reshape(-1)
            ET.SubElement(transform, "matrix", {"value": format_float_list(matrix)})
            return

        if isinstance(value, bool):
            ET.SubElement(
                parent, "boolean", {"name": key, "value": str(value).lower()}
            )
            return

        if isinstance(value, int) and not isinstance(value, bool):
            ET.SubElement(parent, "integer", {"name": key, "value": str(value)})
            return

        if isinstance(value, float):
            ET.SubElement(parent, "float", {"name": key, "value": format_number(value)})
            return

        if isinstance(value, str):
            ET.SubElement(parent, "string", {"name": key, "value": value})
            return

        if isinstance(value, (list, tuple)):
            if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
                tag = "point" if key in POINT_KEYS else "vector"
                ET.SubElement(
                    parent,
                    tag,
                    {
                        "name": key,
                        "x": format_number(value[0]),
                        "y": format_number(value[1]),
                        "z": format_number(value[2]),
                    },
                )
                return

            ET.SubElement(
                parent,
                "string",
                {"name": key, "value": format_float_list(value)},
            )
            return

        raise TypeError(f"Unsupported Mitsuba XML value for {key!r}: {type(value)}")

    def _append_spectrum(self, parent, key, value):
        spec_type = value["type"]
        if spec_type == "rgb":
            rgb = value.get("value", [0.0, 0.0, 0.0])
            ET.SubElement(
                parent,
                "rgb",
                {"name": key, "value": ",".join(format_number(v) for v in rgb[:3])},
            )
            return

        attrs = {"name": key}
        if "filename" in value:
            attrs["filename"] = value["filename"]
        else:
            spec_value = value.get("value", 0.0)
            if isinstance(spec_value, (int, float)):
                attrs["value"] = format_number(spec_value)
            elif spec_value and isinstance(spec_value[0], (tuple, list)):
                attrs["value"] = ", ".join(
                    f"{format_number(w)}:{format_number(s)}" for w, s in spec_value
                )
            else:
                attrs["value"] = format_float_list(spec_value)
        ET.SubElement(parent, "spectrum", attrs)


class WarningSummary:
    def __init__(self):
        self.counts = {}
        self.samples = {}

    def _record(self, key: str, sample: str):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.samples.setdefault(key, sample)

    def log(self, _export_ctx, message, level="INFO"):
        if level == "INFO":
            return

        if level == "ERROR":
            print(f"[mitsuba-export][error] {message}")
            return

        if match := MATERIAL_WARNING_RE.fullmatch(message):
            reason = match.group(1)
            self._record(f"material fallback: {reason}", message)
            return

        if match := UNSUPPORTED_OBJECT_RE.fullmatch(message):
            obj_type = match.group(1)
            self._record(f"unsupported object type: {obj_type}", message)
            return

        if EMPTY_MESH_RE.fullmatch(message):
            self._record("empty meshes skipped", message)
            return

        if SOFT_LIGHT_RE.fullmatch(message):
            self._record("soft shadow radius ignored on lights", message)
            return

        if match := WORLD_WARNING_RE.fullmatch(message):
            self._record(f"world skipped: {match.group(1)}", message)
            return

        print(f"[mitsuba-export][warn] {message}")

    def emit_summary(self):
        if not self.counts:
            return

        print("[mitsuba-export] warning summary:")
        for key, count in sorted(self.counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count}x {key}")


def summarize_scene_dict(scene_data):
    shape_type_counts = Counter()
    mesh_shapes = 0
    light_shapes = 0
    sensors = 0
    emitters = 0

    for value in scene_data.values():
        if not isinstance(value, dict):
            continue

        if "filename" in value and value.get("type") in {"ply", "obj", "serialized"}:
            mesh_shapes += 1
            shape_type_counts[value["type"]] += 1
            continue

        if "emitter" in value and value.get("type") in {
            "rectangle",
            "disk",
            "sphere",
            "cylinder",
        }:
            light_shapes += 1
            shape_type_counts[value["type"]] += 1
            continue

        if value.get("type") in {"perspective", "orthographic", "thinlens"}:
            sensors += 1
            continue

        if value.get("type") in {"point", "spot", "directional", "constant", "envmap"}:
            emitters += 1

    return {
        "mesh_shapes": mesh_shapes,
        "light_shapes": light_shapes,
        "total_shapes": mesh_shapes + light_shapes,
        "shape_types": dict(sorted(shape_type_counts.items())),
        "sensors": sensors,
        "emitters": emitters,
    }


def summarize_xml(path: Path):
    root = ET.parse(path).getroot()
    shape_type_counts = Counter()
    mesh_shapes = 0
    light_shapes = 0
    sensors = 0
    emitters = 0

    for child in root:
        if child.tag == "shape":
            shape_type_counts[child.attrib.get("type")] += 1
            if child.find("emitter") is not None:
                light_shapes += 1
            else:
                mesh_shapes += 1
        elif child.tag == "sensor":
            sensors += 1
        elif child.tag == "emitter":
            emitters += 1

    return {
        "mesh_shapes": mesh_shapes,
        "light_shapes": light_shapes,
        "total_shapes": mesh_shapes + light_shapes,
        "shape_types": dict(sorted(shape_type_counts.items())),
        "sensors": sensors,
        "emitters": emitters,
    }


def print_export_summary(summary, prefix):
    print(
        f"{prefix} shapes: total={summary['total_shapes']} "
        f"mesh={summary['mesh_shapes']} light={summary['light_shapes']} "
        f"sensors={summary['sensors']} emitters={summary['emitters']}"
    )
    print(f"{prefix} shape types: {summary['shape_types']}")


def validate_against_reference(reference_xml: Path, export_summary):
    reference_summary = summarize_xml(reference_xml)
    print_export_summary(reference_summary, "[mitsuba-export] reference")
    if export_summary["mesh_shapes"] != reference_summary["mesh_shapes"]:
        raise RuntimeError(
            "Exported mesh shape count changed: "
            f"{export_summary['mesh_shapes']} != {reference_summary['mesh_shapes']}"
        )
    if export_summary["sensors"] != reference_summary["sensors"]:
        raise RuntimeError(
            "Exported sensor count changed: "
            f"{export_summary['sensors']} != {reference_summary['sensors']}"
        )


def install_write_xml_compat():
    mi = ensure_mitsuba_importable()
    mi.set_variant("scalar_rgb")

    import mitsuba.python as mi_python

    xml_module = types.ModuleType("mitsuba.python.xml")
    xml_module.WriteXML = WriteXMLCompat
    sys.modules["mitsuba.python.xml"] = xml_module
    mi_python.xml = xml_module


def export_scene(
    export_path: Path,
    skip_bake_materials: bool,
    bake_all_materials: bool,
    bake_normal_maps: bool,
    bake_special_maps: bool,
    bake_resolution: int,
    reference_xml: Path | None,
):
    addon_root = prepare_addon_module(ZIP_PATH)
    bake_root = Path(tempfile.mkdtemp(prefix="mitsuba_bake_"))
    changed_world_mappings = []
    try:
        mi = ensure_mitsuba_importable()
        install_write_xml_compat()
        mi.set_log_level(mi.LogLevel.Error)

        materials_module = importlib.import_module(f"{ADDON_MODULE}.io.exporter.materials")
        export_context_module = importlib.import_module(
            f"{ADDON_MODULE}.io.exporter.export_context"
        )
        if not skip_bake_materials:
            bake_scene_materials(
                bake_resolution,
                bake_root,
                materials_module,
                export_context_module,
                bake_all_materials,
                bake_normal_maps,
                bake_special_maps,
            )

        exporter = importlib.import_module(f"{ADDON_MODULE}.io.exporter")
        converter = exporter.SceneConverter()
        warning_summary = WarningSummary()
        converter.export_ctx.log = types.MethodType(warning_summary.log, converter.export_ctx)
        converter.set_path(str(export_path), split_files=False)
        changed_world_mappings = normalize_world_mapping_modes()

        window_manager = bpy.context.window_manager
        depsgraph = bpy.context.evaluated_depsgraph_get()
        window_manager.progress_begin(0, len(depsgraph.object_instances))
        try:
            converter.scene_to_dict(depsgraph, window_manager)
            export_summary = summarize_scene_dict(converter.export_ctx.scene_data)
            print_export_summary(export_summary, "[mitsuba-export] export")
            if reference_xml is not None:
                validate_against_reference(reference_xml, export_summary)
            converter.dict_to_xml()
        finally:
            window_manager.progress_end()
            warning_summary.emit_summary()
    finally:
        restore_world_mapping_modes(changed_world_mappings)
        shutil.rmtree(addon_root, ignore_errors=True)
        shutil.rmtree(bake_root, ignore_errors=True)


def main():
    args = parse_args()
    export_path = Path(args.export_path).expanduser().resolve()
    print(f"Exporting Mitsuba XML to: {export_path}")
    export_scene(
        export_path,
        skip_bake_materials=args.skip_bake_materials,
        bake_all_materials=args.bake_all_materials,
        bake_normal_maps=args.bake_normal_maps,
        bake_special_maps=args.bake_special_maps,
        bake_resolution=args.bake_resolution,
        reference_xml=Path(args.reference_xml).expanduser().resolve()
        if args.reference_xml
        else None,
    )
    print(f"Export complete: {export_path}")


if __name__ == "__main__":
    main()
