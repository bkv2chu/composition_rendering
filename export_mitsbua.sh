#!/bin/bash

set -euxo pipefail

module load blender/4.2.8

run_with_gpu_step() {
  if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[runner] launching under srun to bind a GPU device"
    srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "$@"
    return
  fi
  "$@"
}

DATASET_ROOT=${DATASET_ROOT:-/home/vhchu/composition_rendering/output/random_scenes/orbit_cam_s030411}
SCENE_ID=${SCENE_ID:-000000}

REPO_ROOT=${REPO_ROOT:-/home/vhchu/composition_rendering}
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




# blender -b --python mitsuba/scan_asset_export_cost.py -- \
#     assets/glbs \
#     /scratch/vhchu/asset_export_cost.json
# '
