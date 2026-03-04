#!/bin/bash

set -euxo pipefail
module load opencv/4.12.0

source ~/projects/aip-lindell/vhchu/infinigen/bin/activate
REPO_ROOT=${REPO_ROOT:-/home/vhchu/composition_rendering}
OUT_DIR=${OUT_DIR:-output/random_scenes}
VIDEO_MODE=${VIDEO_MODE:-orbit_cam}
EXPORT_COST_MANIFEST=${EXPORT_COST_MANIFEST:-/scratch/vhchu/asset_export_cost.json}
GLBS_REQUIRE_EXPORT_COST_MANIFEST=${GLBS_REQUIRE_EXPORT_COST_MANIFEST:-true}
GLBS_MAX_BAKED_MESHES=${GLBS_MAX_BAKED_MESHES:-64}
GLBS_MAX_UNIQUE_BAKED_MATERIALS=${GLBS_MAX_UNIQUE_BAKED_MATERIALS:-4}
GLBS_DOWNWEIGHT_EXPORT_COST=${GLBS_DOWNWEIGHT_EXPORT_COST:-true}

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}

find_latest_dataset_root() {
  local out_dir="$1"
  local latest
  latest=$(
    find "$out_dir" -maxdepth 1 -mindepth 1 -type d -name "${VIDEO_MODE}_s*" \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )
  if [ -z "$latest" ]; then
    echo "No composed dataset found under $out_dir for ${VIDEO_MODE}_s*" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

find_latest_scene_id() {
  local dataset_root="$1"
  local latest
  latest=$(
    find "$dataset_root" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended \
      -regex ".*/[0-9]{6}" -printf '%T@ %f\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )
  if [ -z "$latest" ]; then
    echo "No scene directory found under $dataset_root" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}


# python blender_datagen_compose.py \
#   --config configs/render_orbit_cam.yaml \
#   base_path=assets/glbs \
#   out_dir="$OUT_DIR" \
#   num_rendering=1 \
#   glbs_per_scene=20 \
#   shapes_per_scene=0 \
#   num_frames=10 \
#   dump_blend=true \
#   placement_plane_scale=14 \
#   glbs_placement_bbox=[-6.2,-4.45,6.2,4.45] \
#   placement_bbox=[-6.7,-4.95,6.7,4.95] \
#   placement_grid_res=[120,96] \
#   radius_range=[4.5,5.6] \
#   fov_range=[68,84] \
#   camera_object_clearance=1.0 \
#   camera_sample_retry_limit=60 \
#   glbs_z_offset_range=[0.0,0.0] \
#   glbs_max_sample_tries_per_scene=500 \
#   scene_compose_retry_limit=20 \
#   enclosure.enabled=true \
#   enclosure.ceiling=false \
#   enclosure.height=6.0 \
#   glbs_export_cost_manifest="$EXPORT_COST_MANIFEST" \
#   glbs_require_export_cost_manifest="$GLBS_REQUIRE_EXPORT_COST_MANIFEST" \
#   glbs_max_baked_meshes="$GLBS_MAX_BAKED_MESHES" \
#   glbs_max_unique_baked_materials="$GLBS_MAX_UNIQUE_BAKED_MATERIALS" \
#   glbs_downweight_export_cost="$GLBS_DOWNWEIGHT_EXPORT_COST"


deactivate

module load blender/4.2.8

run_with_gpu_step() {
  if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[runner] launching under srun to bind a GPU device"
    srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@"
    return
  fi
  "$@"
}

OUT_DIR_ABS=$(resolve_path "$OUT_DIR")
DATASET_ROOT=${DATASET_ROOT:-$(find_latest_dataset_root "$OUT_DIR_ABS")}
SCENE_ID=${SCENE_ID:-$(find_latest_scene_id "$DATASET_ROOT")}
SCENE_ROOT=${SCENE_ROOT:-"$DATASET_ROOT/$SCENE_ID"}

SCENE_BLEND=${SCENE_BLEND:-"$SCENE_ROOT/scene.blend"}
EXPORT_SCRIPT=${EXPORT_SCRIPT:-"$REPO_ROOT/mitsuba/export_mitsuba.py"}
POSE_SCRIPT=${POSE_SCRIPT:-"$REPO_ROOT/mitsuba/pose_to_npz.py"}
RENDER_SCRIPT=${RENDER_SCRIPT:-"$REPO_ROOT/mitsuba/render_camviews.py"}
EXPORT_XML=${EXPORT_XML:-"$REPO_ROOT/mitsuba_scene/scene.xml"}
RENDER_ROOT=${RENDER_ROOT:-"$REPO_ROOT/mitsuba_scene/renders"}
META_JSON=${META_JSON:-"$SCENE_ROOT/0000.meta.json"}
CAMVIEW_DIR=${CAMVIEW_DIR:-"$SCENE_ROOT/frames/camview/camera_0"}
VARIANT=${VARIANT:-auto}
SPP=${SPP:-1000}
BLENDER_LOG_LEVEL=${BLENDER_LOG_LEVEL:-0}

test -f "$SCENE_BLEND"
test -f "$EXPORT_SCRIPT"
test -f "$POSE_SCRIPT"
test -f "$RENDER_SCRIPT"
test -f "$META_JSON"
mkdir -p "$(dirname "$EXPORT_XML")"
mkdir -p "$RENDER_ROOT"
mkdir -p "$CAMVIEW_DIR"
cd "$REPO_ROOT"

source ~/projects/aip-lindell/vhchu/ts_env/bin/activate
export PYTHONPATH=/home/vhchu/infinigen${PYTHONPATH:+:$PYTHONPATH}

run_with_gpu_step blender \
  --log-level "$BLENDER_LOG_LEVEL" \
  --python-exit-code 1 \
  --python-use-system-env \
  -b "$SCENE_BLEND" \
  -P "$EXPORT_SCRIPT" \
  -- "$EXPORT_XML"

python "$POSE_SCRIPT" "$META_JSON" --output-dir "$CAMVIEW_DIR"

run_with_gpu_step python "$RENDER_SCRIPT" \
  "$EXPORT_XML" \
  "$CAMVIEW_DIR" \
  "$RENDER_ROOT" \
  --variant "$VARIANT" \
  --spp "$SPP" \
  --overwrite

echo "reached the end :)"
