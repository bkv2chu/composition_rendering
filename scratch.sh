module load opencv
python mitsuba/convert_scene_to_confocal.py \
  mitsuba_scene/scene.xml mitsuba_scene/scene_confocal.xml \
  --camview-dir output/random_scenes/orbit_cam_s030417/000000/frames/camview/camera_0 \
  --sensor-width 256 --sensor-height 256 \
  --temporal-bins 2048 --bin-width-opl 0.025 \
  --grid-sigma 0.01 --grid-intensity 1000.0

source ~/projects/aip-lindell/vhchu/ts_env/bin/activate

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENE_DIR="${PROJECT_DIR}/mitsuba_scene/"
NORMAL_SCENE="scene_confocal_camview0001"
INDIRECT_SCENE="$NORMAL_SCENE-nlosonly"
OUTPUT_DIR="/home/vhchu/composition_rendering/mitsuba_scene"


python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/convert_indirect.py "$SCENE_DIR/$NORMAL_SCENE.xml" "$SCENE_DIR/$INDIRECT_SCENE.xml"
python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/render_transient.py "$SCENE_DIR/$NORMAL_SCENE.xml" --clip-max 0.001 --output-dir "$OUTPUT_DIR"
python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/render_transient.py "$SCENE_DIR/$INDIRECT_SCENE.xml" --clip-max 0.001 --output-dir "$OUTPUT_DIR" 
python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/process_transient.py "$OUTPUT_DIR/$NORMAL_SCENE/transient.npy" 
python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/process_transient.py "$OUTPUT_DIR/$INDIRECT_SCENE/transient.npy" 
python /home/vhchu/transient_nvs/fk_torch/perpixel/direct.py "$OUTPUT_DIR/$NORMAL_SCENE/transient_processed.npy" --output_dir "$OUTPUT_DIR" --scene "$SCENE_DIR/$NORMAL_SCENE.xml"
# for LAMBDA_TIMES in 2
#     do
#     python /home/vhchu/transient_nvs/fk_torch/perpixel/ben_phasor_confocal.py \
#         --data  "$OUTPUT_DIR/$INDIRECT_SCENE/transient_processed.npy" \
#         --data-time-axis 2 \
#         --relay-pos "${OUTPUT_DIR}/direct_pos.npy"  \
#         --relay-normals "${OUTPUT_DIR}/direct_normals.npy"  \
#         --scene-file "$SCENE_DIR/$NORMAL_SCENE.xml" \
#         --t0 0.0 \
#         --deltat 0.025 \
#         --spacing-from-relay-pos \
#         --spacing-stat max \
#         --spacing-sample 500000 \
#         --lambda-times $LAMBDA_TIMES \
#         --volume-min -3 -3 -3 \
#         --volume-max  3 3  3 \
#         --voxel-res 256 \
#         --wall-stride 1 \
#         --variant cuda_ad_rgb --projection-axes --save-projections --out "${PROJECT_DIR}/outpp4_lambda${LAMBDA_TIMES}" --cycles 2 --no-relay-weighting
#     done

# python /home/vhchu/transient_nvs/fk_torch/mitransient/scripts/render_transient.py \
#   /home/vhchu/composition_rendering/mitsuba_scene/scene_confocal_camview0000.xml \
#   --clip-max 10.0 \
#   --output-dir 
