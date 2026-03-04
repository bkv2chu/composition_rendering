# # module load opencv/4.12.0
# # source ~/projects/aip-lindell/vhchu/infinigen/bin/activate
# python blender_datagen_compose.py \
#   --config configs/render_orbit_cam.yaml \
#   base_path=assets/glbs \
#   out_dir=output/random_scenes \
#   num_rendering=1 \
#   glbs_per_scene=20 \
#   shapes_per_scene=0 \
#   num_frames=10 \
#   dump_blend=true \
#   placement_plane_scale=14 \
#   glbs_placement_bbox=[-3,-3,3,3] \
#   placement_bbox=[-3.5,-3.5,3.5,3.5] \
#   placement_grid_res=[80,80] \
#   radius_range=[4.0,4.8] \
#   fov_range=[50,60] \
#   glbs_z_offset_range=[0.0,2.0] \
#   glbs_max_sample_tries_per_scene=500 \
#   scene_compose_retry_limit=20 \
#   enclosure.enabled=true \
#   enclosure.ceiling=false \
#   enclosure.height=3.0
# #   glbs_scale_range=[0.3,0.5] \




# Mitsuba export-cost filtering
EXPORT_COST_MANIFEST=${EXPORT_COST_MANIFEST:-/scratch/vhchu/asset_export_cost.json}
GLBS_REQUIRE_EXPORT_COST_MANIFEST=${GLBS_REQUIRE_EXPORT_COST_MANIFEST:-true}
GLBS_MAX_BAKED_MESHES=${GLBS_MAX_BAKED_MESHES:-64}
GLBS_MAX_UNIQUE_BAKED_MATERIALS=${GLBS_MAX_UNIQUE_BAKED_MATERIALS:-4}
GLBS_DOWNWEIGHT_EXPORT_COST=${GLBS_DOWNWEIGHT_EXPORT_COST:-true}

# module load opencv/4.12.0
# source ~/projects/aip-lindell/vhchu/infinigen/bin/activate
python blender_datagen_compose.py \
  --config configs/render_orbit_cam.yaml \
  base_path=assets/glbs \
  out_dir=output/random_scenes \
  num_rendering=1 \
  glbs_per_scene=20 \
  shapes_per_scene=0 \
  num_frames=10 \
  dump_blend=true \
  placement_plane_scale=14 \
  glbs_placement_bbox=[-6.2,-4.45,6.2,4.45] \
  placement_bbox=[-6.7,-4.95,6.7,4.95] \
  placement_grid_res=[120,96] \
  radius_range=[4.5,5.6] \
  fov_range=[68,84] \
  camera_object_clearance=1.0 \
  camera_sample_retry_limit=60 \
  glbs_z_offset_range=[0.0,0.0] \
  glbs_max_sample_tries_per_scene=500 \
  scene_compose_retry_limit=20 \
  enclosure.enabled=true \
  enclosure.ceiling=false \
  enclosure.height=6.0 \
  glbs_export_cost_manifest="$EXPORT_COST_MANIFEST" \
  glbs_require_export_cost_manifest="$GLBS_REQUIRE_EXPORT_COST_MANIFEST" \
  glbs_max_baked_meshes="$GLBS_MAX_BAKED_MESHES" \
  glbs_max_unique_baked_materials="$GLBS_MAX_UNIQUE_BAKED_MATERIALS" \
  glbs_downweight_export_cost="$GLBS_DOWNWEIGHT_EXPORT_COST"
#   glbs_scale_range=[0.3,0.5] \
