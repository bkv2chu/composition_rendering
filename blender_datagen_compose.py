#!/usr/bin/env python3
"""
Blender Data Generation Script for Composition Rendering
Reimplements datagen_compose.py functionality using Blender's Python API
"""

from re import I
import bpy
import bmesh
import math
import random
import json
import os
import shutil
import sys
import logging
import time
import argparse
import glob
import numpy as np
import copy
import torch
import imageio
import imageio.v3 as iio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from mathutils import Vector, Matrix, Quaternion
from pathlib import Path
from shapely.geometry import Polygon
from utils import blender_utils, render_utils, image_utils
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from types import SimpleNamespace

DEPTH_MAX = 1000.0

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s - %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)

# Create a logger
logger = logging.getLogger(__name__)


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        # If seed is None, set seed using current time and process id for randomness
        seed_val = (int(time.time() * 1000) + os.getpid()) % (2**32)
        random.seed(seed_val)
        np.random.seed(seed_val)

def check_msh_bbox(msh):
    vmin, vmax = msh.aabb
    bbox = (vmax - vmin) # [x, y, z]
    hori_ratio = bbox[0] / bbox[2]
    if hori_ratio > 6 or hori_ratio < 1/6:
        return False
    else:
        vert_ratio = (bbox[1] / bbox[0], bbox[1] / bbox[2])
        if max(vert_ratio) < 0.15: # too flat
            return False
        if min(vert_ratio) > 8: # too tall
            return False
    return True

def post_process_rendering(output_dir, feature_fmt='jpg', dump_video=False, video_fps=24):
    blender_passes = ['rgb', 'normal', 'depth', 'albedo', 'orm']
    mask = 0
    for p in blender_passes:
        img_list = sorted(glob.glob(os.path.join(output_dir, f'{p}.*')))
        if p == 'normal' and len(img_list) > 0:
            meta_file = json.load(open(os.path.join(output_dir, f'0000.meta.json')))
            meta_frames = meta_file['frames']
        for img_path in img_list:
            img_basename = os.path.basename(img_path)
            img_name_part = img_basename.split('.')
            pidx, fidx, img_fmt = int(img_name_part[1]), int(img_name_part[2]) - 1, img_name_part[3]
            img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{img_fmt}'
            if p == 'normal':
                w_normal = image_utils.read_normal_exr(img_path)[..., :3] # [H, W, 3]
                mask = (w_normal == 0).all(axis=-1, keepdims=True) # [H, W, 1]
                bg_normal = np.array([0, 0, 1])
                c2w = np.array(meta_frames[fidx]['transform_matrix'])
                w2c_rot = np.linalg.inv(c2w[:3, :3])
                s_normal = w_normal @ w2c_rot.T
                s_normal = s_normal * (1-mask) + mask * bg_normal
                s_normal = (s_normal + 1) * 0.5
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                image_utils.save_image(os.path.join(output_dir, img_new_name), s_normal)
                # remove the original normal
                os.remove(img_path)
            elif p == 'depth':
                depth = image_utils.read_depth_exr(img_path)
                iio.imwrite(os.path.join(output_dir, img_new_name), depth, plugin='opencv')
                os.remove(img_path)
            elif p == 'albedo':
                albedo = image_utils.read_img(img_path)
                img_new_name = f'{pidx:04d}.{fidx:04d}.{p}.{feature_fmt}'
                # NOTE: albedo is in sRGB!!!
                albedo = render_utils.rgb_to_srgb(albedo)
                image_utils.save_image(os.path.join(output_dir, img_new_name), albedo)
                os.remove(img_path)
            elif p == 'orm':
                orm = image_utils.read_img(img_path)
                roughness, metallic = orm[..., 1:2], orm[..., 2:3]
                for key, value, bg_color in zip(['roughness', 'metallic'], [roughness, metallic], [0.5, 0]):
                    img_new_name = f'{pidx:04d}.{fidx:04d}.{key}.{feature_fmt}'
                    value = value * (1-mask) + mask * bg_color
                    image_utils.save_image(os.path.join(output_dir, img_new_name), value[..., 0])
                os.remove(img_path)
            else:
                # os.rename(img_path, os.path.join(output_dir, img_new_name))
                shutil.move(img_path, os.path.join(output_dir, img_new_name))

    # Optionally dump videos from RGB frames grouped by lighting index
    if dump_video:
        # Accept any extension for RGB frames (png/jpg)
        rgb_list = sorted(glob.glob(os.path.join(output_dir, f'*.rgb.*')))
        # Group frames by lighting index (first token)
        group_dict = {}
        for fp in rgb_list:
            base = os.path.basename(fp)
            parts = base.split('.')
            if len(parts) < 4:
                continue
            pidx_str, fidx_str = parts[0], parts[1]
            try:
                pidx = int(pidx_str)
                fidx = int(fidx_str)
            except Exception:
                continue
            group = group_dict.setdefault(pidx, [])
            group.append((fidx, fp))

        for pidx, frames in group_dict.items():
            frames.sort(key=lambda x: x[0])
            if len(frames) < 2:
                continue

            out_path = os.path.join(output_dir, f"{pidx:04d}.rgb.mp4")
            with imageio.get_writer(out_path, fps=float(video_fps)) as writer:
                for _, frame_path in frames:
                    frame = iio.imread(frame_path)
                    if frame is None:
                        continue
                    # Convert float images to uint8
                    if frame.dtype.kind == 'f':
                        frame = np.clip(frame, 0.0, 1.0)
                        frame = (frame * 255.0 + 0.5).astype(np.uint8)
                    elif frame.dtype != np.uint8:
                        # Best-effort cast
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    # Ensure color (H,W,3)
                    if frame.ndim == 2:
                        frame = np.repeat(frame[..., None], 3, axis=-1)
                    writer.append_data(frame)
            logger.info(f"Wrote video: {out_path}")


def render_scene(
    mesh_list, mesh_meta, envlight_path_list, shortname, prefix, FLAGS
):
    num_frames = FLAGS.num_frames
    if FLAGS.video_mode == 'orbit_lgt':
        env_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
    elif FLAGS.video_mode == 'rotat_obj':
        obj_rot_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        mesh_id = [1]
        if mesh_meta is not None and random.random() > 0.5:
            if len(mesh_list) > 2 and 'metallic' not in mesh_meta[2]:
                mesh_id.append(2)
    elif FLAGS.video_mode == 'vtran_obj':
        num_obj = len(mesh_list) - 1
        drop_id = np.random.permutation(num_obj) + 1
        drop_id = drop_id.tolist()
        drop_id = drop_id[:1] if random.random() < 0.5 else drop_id[:2]
        if 1 not in drop_id and random.random() < 0.8:
            drop_id = drop_id + [1]
        num_drop = len(drop_id)
        drop_prev = [0] * num_drop
        drop_range = [0.5, 1.5]
        drop_list = np.random.uniform(*drop_range, size=[num_drop])
        drop_offset_list = []
        bounce = random.random() < 0.5 # always bounce 1/3 of the height
        if bounce:
            bounce_nframes = num_frames // 3 + random.randint(-num_frames//6, num_frames//6)
        else:
            bounce_nframes = 0
        for drop in drop_list:
            drop_offset = np.linspace(drop, 0, num_frames - bounce_nframes, endpoint=True) # the last is 0
            if bounce:
                bounce_factor = random.uniform(0.33, 0.8)
                bounce_offset = np.linspace(drop * bounce_factor, 0, bounce_nframes, endpoint=False)[::-1] # the last might not be 0
                drop_offset = np.concatenate([drop_offset, bounce_offset])
            drop_offset_list.append(drop_offset)

    object_spheres = []
    camera_clearance = float(getattr(FLAGS, 'camera_object_clearance', 0.0) or 0.0)
    if camera_clearance > 0.0:
        for mesh in mesh_list[1:]:
            if mesh is None:
                continue
            vmin, vmax = mesh.aabb
            center = np.array((vmin + vmax) * 0.5, dtype=np.float32)
            radius = float(np.linalg.norm(np.array(vmax - vmin, dtype=np.float32)) * 0.5)
            object_spheres.append((center, radius))

    def sample_camera_path():
        cam_radius = FLAGS.radius_range[0] + np.random.uniform() * (FLAGS.radius_range[1] - FLAGS.radius_range[0])
        fovx = np.deg2rad(FLAGS.fov_range[0] + np.random.uniform() * (FLAGS.fov_range[1] - FLAGS.fov_range[0]))
        azimuth = np.random.uniform(*FLAGS.cam_phi_range)
        elevation = np.random.uniform(*FLAGS.cam_theta_range)
        if FLAGS.cam_t_range is not None:
            t = np.random.uniform(*FLAGS.cam_t_range, size=[3])
        else:
            t = np.zeros(3)

        cam_radius_list = None
        if FLAGS.varying_radius and random.random() < 0.3:
            roll_step = np.random.uniform(-np.pi / 2, np.pi / 2)
            cam_radius_list = FLAGS.radius_range[0] + (FLAGS.radius_range[1] - FLAGS.radius_range[0]) * \
                (1 + np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False) + roll_step)) / 2

        azimuth_offset = None
        elevation_offset = None
        fovx_motion = None
        fovx_fix = None
        radius_fix = None
        if FLAGS.video_mode == 'orbit_cam':
            azimuth_offset = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
            elevation_offset = np.zeros(num_frames)
        elif FLAGS.video_mode == 'oscil_cam':
            phi_center = sum(FLAGS.cam_phi_range) / 2
            phi_ratio = 0.6
            phi_range = (FLAGS.cam_phi_range[1] - FLAGS.cam_phi_range[0]) / 2
            phi_range_clip = phi_range * phi_ratio
            azimuth = np.random.uniform(phi_center - phi_range_clip, phi_center + phi_range_clip)
            cone_angle = np.random.uniform(0.2 * (1 - phi_ratio) * phi_range, phi_range - np.abs(phi_center - azimuth))
            azimuth_offset = np.sin(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
            elevation_offset = np.cos(np.linspace(0, 2*np.pi, num_frames, endpoint=False)) * cone_angle
        elif FLAGS.video_mode == 'dolly_cam':
            fovx_fix = np.deg2rad(45)
            radius_fix = 2.5
            fovx_perturb = np.random.uniform(-10, 10, size=2)
            fovx_motion = np.linspace(
                max(10, FLAGS.fov_range[0] + fovx_perturb[0]),
                FLAGS.fov_range[1] + fovx_perturb[1],
                num_frames,
                endpoint=True,
            )
            if random.random() < 0.5:
                fovx_motion = fovx_motion[::-1]

        azimuth_0, elevation_0 = azimuth, elevation
        cam_list = []
        frame_info_list = []
        fovx_list = [] if FLAGS.video_mode == 'dolly_cam' else None
        for it in range(num_frames):
            azimuth_frame = azimuth_0
            elevation_frame = elevation_0
            radius_frame = cam_radius
            fovx_frame = fovx

            if FLAGS.video_mode in ['orbit_cam', 'oscil_cam']:
                azimuth_frame = azimuth_0 + azimuth_offset[it]
                elevation_frame = elevation_0 + elevation_offset[it]
                elevation_frame = np.clip(elevation_frame, 5 * np.pi / 180, 85 * np.pi / 180)
            elif FLAGS.video_mode == 'dolly_cam':
                fovx_frame = np.deg2rad(fovx_motion[it])
                radius_frame = radius_fix * np.tan(fovx_fix / 2) / np.tan(fovx_frame / 2)
                fovx_list.append(fovx_frame)

            if FLAGS.varying_radius and cam_radius_list is not None and FLAGS.video_mode != 'dolly_cam':
                radius_frame = cam_radius_list[it]

            cam_matrix = blender_utils.get_cam_matrix(azimuth_frame, elevation_frame, t, radius_frame)
            cam_list.append(cam_matrix)
            frame_info = {
                'azimuth': azimuth_frame,
                'elevation': elevation_frame,
            }
            if FLAGS.video_mode == 'dolly_cam':
                frame_info['fov'] = fovx_frame
            frame_info_list.append(frame_info)

        return cam_radius, fovx, t, cam_list, frame_info_list, fovx_list

    def camera_path_has_clearance(cam_list):
        if camera_clearance <= 0.0 or len(object_spheres) == 0:
            return True
        for cam_matrix in cam_list:
            cam_pos = np.array(cam_matrix[:3, 3], dtype=np.float32)
            for center, radius in object_spheres:
                if np.linalg.norm(cam_pos - center) < (radius + camera_clearance):
                    return False
        return True

    camera_retry_limit = int(getattr(FLAGS, 'camera_sample_retry_limit', 25) or 25)
    for camera_try in range(camera_retry_limit):
        cam_radius, fovx, t, cam_list, frame_info_list, fovx_list = sample_camera_path()
        if camera_path_has_clearance(cam_list):
            break
    else:
        raise RuntimeError(
            f"Failed to sample a camera path with object clearance {camera_clearance} "
            f"after {camera_retry_limit} attempts"
        )

    cubemap, vec, vec_ref, latlong_img = None, None, None, None
    if FLAGS.dump_envmap: # TODO:
        vec = render_utils.latlong_vec(FLAGS.resolution)
        vec_ball, mask = render_utils.get_ideal_normal_ball(FLAGS.resolution[0], flip_x=False)
        vec_ref = render_utils.get_ref_vector(vec_ball, np.array([0,0, 1]))
        vec_ref = vec_ref.float() #.to('cuda')
    
    ori_shortname = shortname
    skip_features = False
    dump_format = FLAGS.dump_format # TODO:
    for lgt_i in range(FLAGS.num_lighting):
        prefix = f'{lgt_i:04d}.'
        if FLAGS.prefix_in_folder:
            shortname = f'{ori_shortname}.{lgt_i:04d}'
            prefix = ''
            os.makedirs(os.path.join(FLAGS.out_dir, shortname), exist_ok=True)
            
        skip_features = lgt_i > 0
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            envlight_path = np.random.choice(envlight_path_list, p=FLAGS.envlight_sample_weight)
        
        envmap_strength = np.random.uniform(*FLAGS.random_env_scale) if FLAGS.random_env_scale is not None else FLAGS.env_scale
        envmap_flip = False
        if FLAGS.random_env_flip:
            if random.random() > 0.5:
                envmap_flip = True

        if FLAGS.random_env_rotation:
            envmap_rotation_y = random.uniform(0, 2*np.pi)
        else:
            envmap_rotation_y = 0
        envmap_rotation_y_0 = envmap_rotation_y

        if FLAGS.dump_envmap:
            latlong_img = image_utils.read_img(envlight_path)
            latlong_img = latlong_img * envmap_strength
            latlong_img = np.nan_to_num(latlong_img, nan=0.0, posinf=65504.0, neginf=0.0) 
            latlong_img = np.clip(latlong_img, 0.0, 65504.0)
            latlong_img = torch.tensor(latlong_img, dtype=torch.float32)
            if envmap_flip:
                latlong_img = latlong_img.flip(1)

            # TODO: rotate
            cubemap = render_utils.latlong_to_cubemap_torch(latlong_img, [512, 512])
            env_proj = render_utils.cubemap_sample_torch(cubemap, -vec)
            env_proj = env_proj.flip(0).flip(1)

            env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
            env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_ldr.{dump_format}'), env_ev0)
            image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}env_log.{dump_format}'), env_log)

            if FLAGS.dump_env_bg:
                intrinsic = render_utils.cam_intrinsics(fovx, FLAGS.resolution[1], FLAGS.resolution[0])
                env_uv = render_utils.uv_mesh(FLAGS.resolution[1], FLAGS.resolution[0])
                pos_cam = env_uv @ np.linalg.inv(intrinsic).T

        blender_utils.set_envmap_texture(envlight_path, envmap_rotation_y, envmap_strength, envmap_flip)    
        logger.info(f"EnvProbe {lgt_i}/{FLAGS.num_lighting}: {envlight_path}")

        meta_dict = {
            # 'tone_mapping': FLAGS.tonemap_type,
            # 'envmap': os.path.basename(envlight_path),
            'camera_angle_x': fovx, # fov along width
            'cam_radius': cam_radius,
        }
        if FLAGS.analytical_sky:
            raise NotImplementedError('Not supported yet')
        else:
            meta_dict['envmap'] = os.path.basename(envlight_path)
        meta_frames = []

        # =============================================================================================
        # Start rendering
        # =============================================================================================
        for it in range(num_frames):
            frame_info = frame_info_list[it]
            azimuth = frame_info['azimuth']
            elevation = frame_info['elevation']
            cam_matrix = cam_list[it]

            if FLAGS.video_mode == 'orbit_lgt':
                envmap_rotation_y = envmap_rotation_y_0 + env_rot_offset[it]
                # blender_utils.rotate_envmap(envmap_rotation_y) # TODO:
                        
            elif FLAGS.video_mode == 'vtran_obj':
                pass

            if FLAGS.dump_envmap:
                c2w = render_utils.convert_cam_mat_blender_to_dr(cam_matrix)
                c2w = torch.tensor(c2w, dtype=torch.float32, device=vec.device)
                vec_cam = vec.reshape(-1, 3) @ c2w[:3, :3].T
                y_rot = render_utils.rotate_y(envmap_rotation_y, device=vec.device)
                vec_query = (vec_cam @ y_rot[:3, :3].T).reshape(1, *FLAGS.resolution, 3)
                env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                env_proj = env_proj.flip(0).flip(1)
                env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_ldr.{dump_format}'), env_ev0)
                image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_log.{dump_format}'), env_log)

                if FLAGS.dump_ball_env:
                    vec_ball = -vec_ref.reshape(-1, 3) @ c2w[:3, :3].T
                    vec_query = (vec_ball @ y_rot[:3, :3].T).reshape(1, FLAGS.resolution[0], FLAGS.resolution[0], 3)
                    env_proj = render_utils.cubemap_sample_torch(cubemap, -vec_query)[0]
                    env_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(env_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    env_log = render_utils.rgb_to_srgb(torch.log1p(env_proj) / np.log1p(10000)).clip(0, 1).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_ldr.{dump_format}'), env_ev0)
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.ball_env_log.{dump_format}'), env_log)

                if FLAGS.dump_env_bg:
                    bg_dir = pos_cam @ c2w[:3, :3].T
                    bg_dir = (bg_dir @ y_rot[:3, :3].T)
                    bg_q_dir = -bg_dir.flip(1).contiguous().reshape(1, *FLAGS.resolution, 3)
                    bg_proj = render_utils.cubemap_sample_torch(cubemap, bg_q_dir)[0]
                    bg_ev0 = render_utils.rgb_to_srgb(render_utils.reinhard(bg_proj, max_point=16).clip(0, 1)).cpu().numpy()
                    image_utils.save_image(os.path.join(FLAGS.out_dir, f'{shortname}/{prefix}{it:04d}.env_bg.{dump_format}'), bg_ev0)



            meta_frame = {
                # camera attributes
                'transform_matrix': cam_matrix.tolist(), # standard blender c2w
                'elevation': elevation,
                'azimuth': azimuth,
                # envmap
                'envmap_rot': envmap_rotation_y,
                'envmap_strength': envmap_strength,
                'envmap_flip': envmap_flip,
            }
            if FLAGS.video_mode == 'rotat_obj':
                meta_frame['obj_rot'] = obj_rot_offset[it]
                meta_frame['obj_rot_id'] = mesh_id
            if FLAGS.video_mode == 'vtran_obj':
                meta_frame['drop_offset'] = [drop_offset_list[i][it] for i in range(num_drop)]
                meta_frame['drop_id'] = drop_id
            if FLAGS.video_mode == 'dolly_cam':
                meta_frame['fov'] = frame_info['fov']

            meta_frames.append(meta_frame)

        if not skip_features or lgt_i == 0: # only setup the camera update for the first lighting setup
            blender_utils.setup_realtime_camera_update(cam_list, cam_mode='MATRIX', fov_sequence=fovx_list)
            if FLAGS.video_mode == 'rotat_obj':
                for mi in mesh_id:
                    init_rot = mesh_list[mi].empty.rotation_euler[2]
                    mesh_list[mi].setup_realtime_update(num_frames, rotation=obj_rot_offset + init_rot)
            elif FLAGS.video_mode == 'vtran_obj':
                for di,mi in enumerate(drop_id):
                    init_loc = mesh_list[mi].empty.location
                    drop_offset_vec3 = np.zeros((num_frames, 3))
                    drop_offset_vec3[:, 2] = drop_offset_list[di]
                    drop_offset_vec3 += init_loc
                    mesh_list[mi].setup_realtime_update(num_frames, translation=drop_offset_vec3)
        
        if FLAGS.video_mode == 'orbit_lgt': # on orbit lgt needs reset.
            envmap_rotation_y_list = envmap_rotation_y_0 + env_rot_offset
            blender_utils.setup_realtime_envmap_update(envmap_rotation_y_list.tolist())

        # set up the rendering
        save_folder = os.path.join(FLAGS.out_dir, shortname)
        blender_utils.setup_camera_settings(
            resolution_x=FLAGS.resolution[1], resolution_y=FLAGS.resolution[0], fov_rad=fovx,
        )
        blender_utils.setup_cycles_rendering(samples=FLAGS.spp, use_denoise=FLAGS.use_denoise, transparent_bg=FLAGS.transparent_bg)

        blender_passes = ['rgb']
        if not skip_features:
            blender_utils.setup_render_passes(['normal', 'depth', 'diffcol', 'object', 'material'])
            blender_utils.setup_compositor_nodes(output_dir=save_folder, 
                passes=['normal', 'depth'], suffix=f'.{0:04d}') # NOTE: ['rgb', 'diffcol'] is removed here
            blender_utils.render_albedo_and_material(output_dir=save_folder, passes=['albedo', 'orm'], suffix=f'.{0:04d}')
            blender_passes.extend(['normal', 'depth', 'albedo', 'orm'])
        # else:
        #     blender_utils.setup_compositor_nodes(output_dir=save_folder, passes=['rgb'], suffix=f'.{lgt_i:04d}')
        blender_utils.render_all_frames(output_dir=save_folder, num_frames=num_frames, suffix=f'rgb.{lgt_i:04d}')
        
        # dump the meta data
        meta_dict['frames'] = meta_frames
        meta_dict['file_path'] = shortname
        with open(os.path.join(FLAGS.out_dir, shortname, f'{prefix}meta.json'), 'w') as f:
            _meta_dict = copy.deepcopy(meta_dict)
            _meta_dict['mesh_list'] = mesh_meta
            json.dump(_meta_dict, f, indent=4)

class GLTFFileManger:
    def __init__(
        self, files, random_sample=True, rescale=True, 
        multi_sample_weight=None, check_bbox=False, file_weights=None
    ):
        # just assume files is a list of list of files
        self.files = files
        self.num_file_lists = len(files)
        self.random_sample = random_sample
        self.rescale = rescale
        self.multi_sample_weight = multi_sample_weight
        self.check_bbox = check_bbox
        self.file_weights = None
        if self.multi_sample_weight is not None:
            assert len(self.multi_sample_weight) == self.num_file_lists
            self.multi_sample_weight = np.array(self.multi_sample_weight, dtype=np.float64)
        if file_weights is not None:
            assert len(file_weights) == self.num_file_lists
            self.file_weights = [
                np.array(weights, dtype=np.float64) if weights is not None else None
                for weights in file_weights
            ]
        self._rebuild_files_list()
        self._normalize_file_weights()
            
    def __len__(self):
        return self.num_files

    def __iter__(self):
        self.idx = 0
        return self

    def _rebuild_files_list(self):
        self.files_list = []
        for file_group in self.files:
            self.files_list.extend(file_group)
        self.num_files = len(self.files_list)

    def _normalize_file_weights(self):
        if self.file_weights is None:
            return
        for i, weights in enumerate(self.file_weights):
            if weights is None:
                continue
            if len(weights) != len(self.files[i]):
                raise ValueError("file_weights must match files layout")
            if len(weights) == 0:
                continue
            total = float(weights.sum())
            if total <= 0:
                weights = np.ones(len(weights), dtype=np.float64)
                total = float(weights.sum())
            self.file_weights[i] = weights / total

    def _list_sampling_weights(self):
        active = np.array([len(group) > 0 for group in self.files], dtype=np.float64)
        if active.sum() == 0:
            raise StopIteration("No asset files remaining")
        if self.multi_sample_weight is None:
            return active / active.sum()
        weights = self.multi_sample_weight.copy()
        weights *= active
        total = float(weights.sum())
        if total <= 0:
            return active / active.sum()
        return weights / total

    def _find_file_indices(self, file):
        for list_idx, file_group in enumerate(self.files):
            for file_idx, candidate in enumerate(file_group):
                if candidate == file:
                    return list_idx, file_idx
        return None, None

    def _remove_file(self, list_idx, file_idx):
        del self.files[list_idx][file_idx]
        if self.file_weights is not None and self.file_weights[list_idx] is not None:
            self.file_weights[list_idx] = np.delete(self.file_weights[list_idx], file_idx)
        self._rebuild_files_list()
        self._normalize_file_weights()

    def __next__(self): # inifi-gen
        self.idx += 1
        sample_success = False
        obj_mesh = None
        while not sample_success:
            if self.num_files == 0:
                raise StopIteration("No asset files remaining")
            idx = None
            file_idx = None
            if self.random_sample:
                idx = int(np.random.choice(self.num_file_lists, p=self._list_sampling_weights()))
                if self.file_weights is not None and self.file_weights[idx] is not None:
                    file_idx = int(np.random.choice(len(self.files[idx]), p=self.file_weights[idx]))
                else:
                    file_idx = int(np.random.choice(len(self.files[idx])))
                file = self.files[idx][file_idx]
            else:
                file = self.files_list[(self.idx-1) % self.num_files]
                idx, file_idx = self._find_file_indices(file)
                sample_success = True
            try:
                logger.info(f'Processing: {file}')
                _obj_mesh = blender_utils.add_object_file(
                    file, with_empty=True, recenter=True, rescale=self.rescale
                )
                if self.check_bbox and not check_msh_bbox(_obj_mesh):
                    _obj_mesh.clear_objects()
                    del _obj_mesh
                    raise Exception('Invalid bbox')
                else:
                    obj_mesh = _obj_mesh
                    sample_success = True
            except Exception as e:
                logger.info(f'---> Error: {e}, skipping file {file}')
                if idx is not None and file_idx is not None:
                    self._remove_file(idx, file_idx)
                if self.num_files == 0:
                    raise StopIteration("No asset files remaining after filtering failures")
        
        return {'mesh': obj_mesh, 'name': file}


def load_asset_export_cost_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    manifest = {}
    for entry in entries:
        asset_path = entry.get("asset_path") or entry.get("name")
        if not asset_path:
            continue
        manifest[canonicalize_asset_path(asset_path)] = entry
    return manifest


def canonicalize_asset_path(path):
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def asset_export_cost_rejection_reason(entry, flags):
    if entry is None:
        if FLAGS_VALUE(flags, "glbs_require_export_cost_manifest", False):
            return "missing manifest entry"
        return None

    if entry.get("status", "ok") != "ok":
        return f"manifest status={entry.get('status')}"

    checks = [
        ("glbs_max_mesh_objects", "mesh_objects", "mesh_objects"),
        ("glbs_max_baked_meshes", "baked_mesh_objects", "baked_mesh_objects"),
        (
            "glbs_max_unique_baked_materials",
            "unique_baked_materials",
            "unique_baked_materials",
        ),
        ("glbs_max_baked_ratio", "baked_object_ratio", "baked_object_ratio"),
    ]
    for flag_name, entry_key, label in checks:
        limit = FLAGS_VALUE(flags, flag_name, None)
        value = entry.get(entry_key)
        if limit is not None and value is not None and value > limit:
            return f"{label}={value} exceeds {limit}"
    return None


def compute_asset_export_cost_weight(entry, flags):
    if entry is None or entry.get("status", "ok") != "ok":
        return 1.0
    if not FLAGS_VALUE(flags, "glbs_downweight_export_cost", False):
        return 1.0

    baked_meshes = float(entry.get("baked_mesh_objects", 0))
    unique_baked_materials = float(entry.get("unique_baked_materials", 0))
    mesh_objects = float(entry.get("mesh_objects", 0))
    score = (
        baked_meshes
        + float(FLAGS_VALUE(flags, "glbs_export_cost_material_penalty", 25.0))
        * unique_baked_materials
        + float(FLAGS_VALUE(flags, "glbs_export_cost_mesh_penalty", 0.0)) * mesh_objects
    )
    scale = float(FLAGS_VALUE(flags, "glbs_export_cost_weight_scale", 0.01))
    min_weight = float(FLAGS_VALUE(flags, "glbs_export_cost_weight_min", 0.05))
    weight = math.exp(-scale * score)
    return max(min_weight, weight)


def filter_asset_files_by_export_cost(obj_files, manifest, flags):
    filtered_files = []
    file_weights = []
    kept = 0
    dropped = []
    missing = 0

    for file_group in obj_files:
        filtered_group = []
        weights_group = []
        for file in file_group:
            asset_path = canonicalize_asset_path(file)
            entry = manifest.get(asset_path)
            reason = asset_export_cost_rejection_reason(entry, flags)
            if reason is not None:
                dropped.append((asset_path, reason))
                continue
            if entry is None:
                missing += 1
            filtered_group.append(asset_path)
            weights_group.append(compute_asset_export_cost_weight(entry, flags))
            kept += 1
        filtered_files.append(filtered_group)
        file_weights.append(weights_group)

    total = sum(len(group) for group in obj_files)
    logger.info(
        f"Asset export-cost manifest kept {kept}/{total} assets, "
        f"dropped {len(dropped)}, missing entries {missing}"
    )
    if missing:
        logger.warning(
            "Asset export-cost manifest missed %d asset paths after canonicalization",
            missing,
        )
    for asset_path, reason in dropped[:10]:
        logger.info(f"Manifest drop: {asset_path} ({reason})")

    if kept == 0:
        raise RuntimeError("All assets were filtered out by glbs_export_cost_manifest settings")

    return filtered_files, file_weights


def FLAGS_VALUE(flags, key, default):
    value = getattr(flags, key, default)
    return default if value is None else value
    
def main():
    # Parse --config and support legacy flags; additional overrides use OmegaConf dotlist (key=val)
    parser = argparse.ArgumentParser(description='composition_rendering')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    # Legacy convenience flags retained for compatibility
    parser.add_argument('-n', '--num_frames', type=int, default=None)
    parser.add_argument('-o', '--out_dir', type=str, default=None)
    parser.add_argument('--base_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args, unknown = parser.parse_known_args()

    # Defaults
    default_cfg = {
        'seed': None,
        'out_dir': '.',
        'base_path': None,
        'cam_near_far': [0.1, 1000.0],
        'resolution': [256, 256],
        'baseshape_path': None,
        'envlight': 'data/envmap/aerodynamics_workshop_512.hdr',
        'env_scale': 1.0,
        'env_res': [256, 512],
        'probe_res': 256,
        'spp': 8,
        'use_denoise': 'OPTIX',
        'transparent_bg': True,
        'radius_range': [2.0, 4.0],
        'varying_radius': False,
        'camera_object_clearance': 0.0,
        'camera_sample_retry_limit': 25,
        'num_files': None,
        'random_env_rotation': True,
        'random_env_flip': True,
        'random_env_scale': None,
        'bg_color': [1.0, 1.0, 1.0],
        'bg_features': [0.0, 0.0, 0.0],
        'dump_env_bg': False,
        'dump_alpha': False,
        'ray_depth': 1,
        'num_workers': 0,
        'timeout': -1,
        'fov_range': [45.0, 45.0],
        'dump_features': False,
        'num_rendering': 10,
        'num_lighting': 1,
        'cam_phi_range': [0, 360],
        'cam_theta_range': [0, 90],
        'cam_t_range': [0, 0],
        'dump_shading': False,
        'dump_splitsum': False,
        'dump_envmap': False,
        'dump_ball_env': False,
        'dump_irradiance': False,
        'dump_format': 'jpg',
        'dump_video': False,
        'dump_complete': False,
        'dump_blend': False,
        'dump_placement': False,
        'video_mode': 'orbit_cam',
        # Sampling config
        'glbs_check_bbox': False,
        'glbs_multi_sample_weight': None,
        'glbs_rescale': True,
        'glbs_random_sample': True,
        'glbs_per_scene': 2,
        'glbs_max_sample_tries_per_scene': None,
        'glbs_export_cost_manifest': None,
        'glbs_require_export_cost_manifest': False,
        'glbs_max_mesh_objects': None,
        'glbs_max_baked_meshes': None,
        'glbs_max_unique_baked_materials': None,
        'glbs_max_baked_ratio': None,
        'glbs_downweight_export_cost': False,
        'glbs_export_cost_weight_scale': 0.01,
        'glbs_export_cost_weight_min': 0.05,
        'glbs_export_cost_material_penalty': 25.0,
        'glbs_export_cost_mesh_penalty': 0.0,
        'glbs_scale_range': [1.0, 1.5],
        'glbs_rotation_range': [-45, 45],
        'glbs_placement_bbox': [-1.0, -1.0, 1.0, 1.0],
        'glbs_z_offset_range': None,
        'scene_compose_retry_limit': 10,
        'shapes_per_scene': 3,
        'shapes_scale_range': [0.3, 0.8],
        'shapes_rotation_range': [-45, 45],
        'shapes_placement_bbox': [-1.5, -1.5, 1.5, 1.5],
        'placement_centered': False,
        'placement_bbox': [-1.5, -1.5, 1.5, 1.5],
        'placement_grid_res': [40, 40],
        'placement_bbox_scale': 1.1,
        'placement_plane_offset': [0, 0, -0.5],
        'placement_plane_scale': 10,
        'placement_plane': 'data/plane_basic/plane.glb',
        'plane_sample_weight': None,
        'envlight_sample_weight': None,
        'texture_sample_weight': None,
        'placement_plane_textures': None,
        'enclosure': {
            'enabled': False,
            'size': None,
            'height': 2.0,
            'thickness': 0.1,
            'floor_thickness': 0.1,
            'ceiling': False,
            'procedural_floor': True,
            'seam_epsilon': 0.001,
            'color': [0.85, 0.85, 0.85],
            'roughness': 0.8,
            'metallic': 0.0,
        },
        'prefix_in_folder': False,
        # Unsupported/unused but kept for completeness
        'analytical_sky': False,
        'use_objaverse': False,
        'objaverse_selection': None,
        'num_frames': 8,
        'sample_shape_texture': False,
    }

    cfg: DictConfig = OmegaConf.create(default_cfg)
    if args.config is not None:
        file_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, file_cfg)
    # Dotlist CLI overrides (e.g., num_frames=8 out_dir=output/ path.with.dots=value)
    if len(unknown) > 0:
        dotlist = [tok for tok in unknown if '=' in tok]
        if len(dotlist) > 0:
            cli_cfg = OmegaConf.from_cli(dotlist)
            cfg = OmegaConf.merge(cfg, cli_cfg)

    # Apply legacy flags if provided
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.num_frames is not None:
        cfg.num_frames = int(args.num_frames)
    if args.base_path is not None:
        cfg.base_path = args.base_path
    if args.seed is not None:
        cfg.seed = int(args.seed)

    # Convert to plain python containers to safely mutate later
    _flags_container = OmegaConf.to_container(cfg, resolve=True)
    FLAGS = SimpleNamespace(**_flags_container)

    logger.info('Config / Flags (OmegaConf):')
    logger.info('---------')
    logger.info('\n' + OmegaConf.to_yaml(cfg))
    logger.info('---------')

    if FLAGS.seed is not None:
        sub_folder = f"s{FLAGS.seed:06d}"         
        set_seed(FLAGS.seed)
    else:
        set_seed(None)
        sub_folder = 's' + time.strftime("%m%d%H")

    sub_folder = f"{FLAGS.video_mode}_{sub_folder}"
    FLAGS.out_dir = os.path.join(FLAGS.out_dir, sub_folder)
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    try:
        bpy.context.preferences.filepaths.use_relative_paths = False
    except Exception:
        pass

    FLAGS.cam_phi_range = [float(np.deg2rad(float(x)) + np.pi) for x in list(FLAGS.cam_phi_range)]
    FLAGS.cam_theta_range = [float(np.deg2rad(float(x))) for x in list(FLAGS.cam_theta_range)]
    assert FLAGS.video_mode in ['orbit_cam', 'oscil_cam', 'orbit_lgt', 'rotat_obj', 'vtran_obj', 'dolly_cam', 'drop_phy']

    # Composition Sampling Related
    glbs_placement_vmin, glbs_placement_vmax = np.array(FLAGS.glbs_placement_bbox[:2]), np.array(FLAGS.glbs_placement_bbox[2:])
    shapes_placement_vmin, shapes_placement_vmax = np.array(FLAGS.shapes_placement_bbox[:2]), np.array(FLAGS.shapes_placement_bbox[2:])
    bbox_scale = 0.5 * FLAGS.placement_bbox_scale

    placement_vmin, placement_vmax = np.array(FLAGS.placement_bbox[:2]), np.array(FLAGS.placement_bbox[2:])
    placement_range = placement_vmax - placement_vmin
    placement_grid = np.zeros(FLAGS.placement_grid_res, dtype=np.int32)
    placement_bbox2grid = np.array(FLAGS.placement_grid_res) / placement_range
    placement_plane_offset = np.array(FLAGS.placement_plane_offset, dtype=np.float32)
    placement_plane_scale = np.array(FLAGS.placement_plane_scale, dtype=np.float32)
    placement_plane_path = FLAGS.placement_plane

    if placement_plane_path is None:
        placement_plane_path = []
    elif os.path.isdir(placement_plane_path):
        placement_plane_path = sorted([os.path.abspath(p) for p in glob.glob(placement_plane_path + "/*.glb")])
    else:
        placement_plane_path = [os.path.abspath(placement_plane_path)]

    num_planes = len(placement_plane_path)
    plane_sample_weight = np.ones(num_planes)
    if FLAGS.plane_sample_weight is not None:
        for i, plane in enumerate(placement_plane_path):
            for k, w in FLAGS.plane_sample_weight.items():
                if k in plane:
                    plane_sample_weight[i] = w
    FLAGS.plane_sample_weight = plane_sample_weight / plane_sample_weight.sum()

    def sample_glb(data_iter, obj_idx=0):
        target = next(data_iter) # centered, scaled
        ref_mesh = target['mesh']
        mesh_name = target['name']

        if ref_mesh is None:
            logger.info(f'Mesh {mesh_name} is None')
            return None
        num_materials = ref_mesh.get_num_materials()
        if num_materials == 0:
            logger.info(f'Mesh {mesh_name} has invalid materials {num_materials}')
            ref_mesh.clear_objects()
            del ref_mesh
            return None

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.glbs_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.glbs_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        mesh_center = np.array((vmax + vmin) * 0.5)
        cz = (mesh_center[2] - vmin[2])
        z_offset = 0.0
        if FLAGS.glbs_z_offset_range is not None:
            z_min, z_max = [float(v) for v in FLAGS.glbs_z_offset_range]
            if z_min > z_max:
                raise ValueError(f"Invalid glbs_z_offset_range: {FLAGS.glbs_z_offset_range}")
            effective_z_max = z_max
            if FLAGS.enclosure is not None and FLAGS.enclosure.get('enabled', False):
                room_height = float(FLAGS.enclosure.get('height', 2.0))
                mesh_height = float(vmax[2] - vmin[2])
                effective_z_max = min(effective_z_max, max(0.0, room_height - mesh_height))
            if effective_z_max < z_min:
                logger.info(
                    f'Mesh {mesh_name} cannot satisfy z-offset range {FLAGS.glbs_z_offset_range}'
                )
                ref_mesh.clear_objects()
                del ref_mesh
                return None
            z_offset = float(np.random.uniform(z_min, effective_z_max))
            cz = cz + z_offset
        mesh_bounds = np.array(vmax - vmin)[:2]
        if FLAGS.video_mode == 'rotat_obj':
            # use square bbox
            mesh_bounds[:] = np.max(mesh_bounds)
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        
        mesh_placement_vmin =  glbs_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  glbs_placement_vmax - mesh_bounds * 0.5

        if not FLAGS.placement_centered:
            if np.any(mesh_placement_vmin > mesh_placement_vmax):
                logger.info(f'Mesh {mesh_name} is too large for placement bbox')
                ref_mesh.clear_objects()
                del ref_mesh
                return None
            # Sample the object placement
            find_placement = False
            for t in range(5): # 8 tries
                # sample the center
                cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
                cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
                # convert to grid
                gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
                gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
                # check if the placement is valid
                x_lb, x_ub = max(gx-bx, 0), gx+bx
                y_lb, y_ub = max(gy-by, 0), gy+by
                if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                    find_placement = True
                    placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                    break
                
            if not find_placement:
                logger.error(f'\033[91mCannot find valid placement for {mesh_name}\033[0m')
                ref_mesh.clear_objects()
                del ref_mesh
                return None
        else:
            cx, cy = 0, 0
            cz = 0 if FLAGS.glbs_rescale else mesh_center[2]

        # logger.info('cx, cy, cz, smpl_scale', cx, cy, cz, smpl_scale)
        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'num_materials': num_materials,
            'z_offset': z_offset,
        }

        return ref_mesh, placement_meta

    def sample_shape(shapes_files, obj_idx=0):
        target_file = np.random.choice(shapes_files)
        ref_mesh = blender_utils.add_object_file(target_file, with_empty=True, recenter=True, rescale=True)
        mesh_name = target_file

        # Sample the object rotation
        smpl_rot_deg = np.random.uniform(*FLAGS.shapes_rotation_range)
        ref_mesh.apply_transform((0, 0, 0), rotation=np.deg2rad(smpl_rot_deg))
        # Sample the object scale
        smpl_scale = np.random.uniform(*FLAGS.shapes_scale_range)
        vmin, vmax = ref_mesh.aabb
        vmin, vmax = vmin * smpl_scale, vmax * smpl_scale
        mesh_center = np.array((vmax + vmin) * 0.5)
        cz = (mesh_center[2] - vmin[2])
        mesh_bounds = np.array(vmax - vmin)[:2]
        mesh_gbounds = mesh_bounds * placement_bbox2grid
        bx, by = int(np.ceil(mesh_gbounds[0] * bbox_scale)), int(np.ceil(mesh_gbounds[1] * bbox_scale))
        mesh_placement_vmin =  shapes_placement_vmin + mesh_bounds * 0.5
        mesh_placement_vmax =  shapes_placement_vmax - mesh_bounds * 0.5

        # Sample the object placement
        find_placement = False
        for t in range(8): # 8 tries
            # sample the center
            cx = np.random.uniform(mesh_placement_vmin[0], mesh_placement_vmax[0])
            cy = np.random.uniform(mesh_placement_vmin[1], mesh_placement_vmax[1])
            # convert to grid
            gx = round(((cx - placement_vmin[0]) * placement_bbox2grid[0]).item())
            gy = round(((cy - placement_vmin[1]) * placement_bbox2grid[1]).item())
            # check if the placement is valid
            x_lb, x_ub = max(gx-bx, 0), gx+bx
            y_lb, y_ub = max(gy-by, 0), gy+by
            if not placement_grid[x_lb:x_ub, y_lb:y_ub].any():
                find_placement = True
                placement_grid[x_lb:x_ub, y_lb:y_ub] += (obj_idx + 1)
                break

        if not find_placement:
            logger.info(f'Cannot find valid placement for {mesh_name}')
            ref_mesh.clear_objects()
            del ref_mesh
            return None

        smpl_translation = np.array((cx, cy, cz))
        smpl_translation = smpl_translation - mesh_center + placement_plane_offset
        ref_mesh.apply_transform(smpl_translation, scale=smpl_scale)

        # Material sampling
        roughness_range = [0, 0.8]
        roughness = np.random.uniform(roughness_range[0], roughness_range[1]) ** 2
        metallic = np.abs(np.random.normal(0, 0.25))
        # 50% metallic (close to 1)
        if np.random.uniform() < 0.5:
            metallic = 1 - metallic
        if metallic > 0.5:
            base_color = np.random.randint(170, 255, size=3) / 255
        else:
            base_color = np.random.randint(30, 240, size=3) / 255

        logger.info(f'shape material: {base_color}, {roughness}, {metallic}')
        ref_mesh.set_principled_material(base_color, roughness, metallic)

        placement_meta = {
            'name': mesh_name,
            'translation': smpl_translation.tolist(),
            'rotation': smpl_rot_deg,
            'scale': smpl_scale,
            'roughness': roughness,
            'metallic': metallic,
            'base_color': base_color.tolist()
        }

        return ref_mesh, placement_meta

    # plane textures
    plane_textures_path = []
    if FLAGS.placement_plane_textures is not None:
        plane_textures_path = sorted([
            os.path.abspath(p) for p in glob.glob(os.path.join(FLAGS.placement_plane_textures, '*'))
        ])
        num_plane_textures = len(plane_textures_path)
        texture_sample_weight = np.ones(num_plane_textures)
        if FLAGS.texture_sample_weight is not None:
            for i, texture in enumerate(plane_textures_path):
                for k, w in FLAGS.texture_sample_weight.items():
                    if k in texture:
                        texture_sample_weight[i] = w
        FLAGS.texture_sample_weight = texture_sample_weight / texture_sample_weight.sum()
        
    obj_files = [] 
    if not FLAGS.use_objaverse and FLAGS.base_path is not None:
        base_path_abs = os.path.abspath(FLAGS.base_path)
        if not os.path.isfile(base_path_abs):
            obj_files.append([os.path.abspath(p) for p in (glob.glob(os.path.join(base_path_abs, "*.glb")) + \
                glob.glob(os.path.join(base_path_abs, "*.gltf")) + \
                glob.glob(os.path.join(base_path_abs, "*.obj")))])
        else:
            obj_files.append([os.path.abspath(p) for p in np.loadtxt(base_path_abs, dtype=str).tolist()])
    else:
        raise NotImplementedError("Objaverse is not supported yet")

    file_weights = None
    if FLAGS.glbs_export_cost_manifest is not None:
        manifest_path = os.path.abspath(FLAGS.glbs_export_cost_manifest)
        manifest = load_asset_export_cost_manifest(manifest_path)
        obj_files, file_weights = filter_asset_files_by_export_cost(obj_files, manifest, FLAGS)

   
    # obj dataloader
    obj_dataloader = GLTFFileManger(
        obj_files, 
        random_sample=FLAGS.glbs_random_sample, 
        rescale=FLAGS.glbs_rescale, 
        multi_sample_weight=FLAGS.glbs_multi_sample_weight, 
        check_bbox=FLAGS.glbs_check_bbox,
        file_weights=file_weights,
    )
    logger.info(f"Use {len(obj_dataloader)} glbs")
    if FLAGS.glbs_max_sample_tries_per_scene is None:
        FLAGS.glbs_max_sample_tries_per_scene = max(FLAGS.glbs_per_scene * 25, 50)
    logger.info(
        f"Will place exactly {FLAGS.glbs_per_scene} GLBs per scene with up to "
        f"{FLAGS.glbs_max_sample_tries_per_scene} GLB samples per attempt and "
        f"{FLAGS.scene_compose_retry_limit} scene retries"
    )

    # envlight 
    env_src = os.path.abspath(FLAGS.envlight)
    if not os.path.isfile(env_src):
        envlight_path_list = sorted([os.path.abspath(p) for p in (glob.glob(os.path.join(env_src, "*.exr"))  + \
            glob.glob(os.path.join(env_src, "*.hdr")))])
    elif env_src.endswith('.txt'):
        envlight_path_list = [os.path.abspath(p) for p in np.loadtxt(env_src, dtype=str).tolist()]
    else:
        envlight_path_list = [env_src]

    num_envlights = len(envlight_path_list)
    envlight_sample_weight = np.ones(num_envlights)
    if FLAGS.envlight_sample_weight is not None:
        for i, envlight in enumerate(envlight_path_list):
            for k, w in FLAGS.envlight_sample_weight.items():
                if k in envlight:
                    envlight_sample_weight[i] = w
    FLAGS.envlight_sample_weight = envlight_sample_weight / envlight_sample_weight.sum()

    baseshape_files = []
    if FLAGS.baseshape_path is not None:
        bsp = os.path.abspath(FLAGS.baseshape_path)
        if os.path.isdir(bsp):
            baseshape_files = [os.path.abspath(p) for p in glob.glob(os.path.join(bsp, "*.glb"))]
        else:
            baseshape_files = [bsp]

    def create_principled_material(name, base_color, roughness, metallic):
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        principled.inputs['Base Color'].default_value = (
            float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0
        )
        principled.inputs['Roughness'].default_value = float(roughness)
        principled.inputs['Metallic'].default_value = float(metallic)
        return material

    def create_diffuse_material(name, base_color):
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)
        diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        diffuse.location = (0, 0)
        links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
        diffuse.inputs['Color'].default_value = (
            float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0
        )
        return material

    def add_box_primitive(name, size_xyz, location, material, flip_normals=False):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = Vector((size_xyz[0], size_xyz[1], size_xyz[2]))
        bpy.context.view_layer.update()
        if flip_normals:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.materials.clear()
        obj.data.materials.append(material)
        return obj

    def add_floor_slab(
        scene_name,
        size_xy,
        center_xy,
        top_z,
        thickness,
        texture_path=None,
        texture_scale=None,
        source_name=None,
    ):
        enclosure_cfg = FLAGS.enclosure if FLAGS.enclosure is not None else {}
        floor_color = enclosure_cfg.get('color', [0.85, 0.85, 0.85])
        floor_roughness = float(enclosure_cfg.get('roughness', 0.8))
        floor_metallic = float(enclosure_cfg.get('metallic', 0.0))
        floor_material_kind = str(enclosure_cfg.get('floor_material', 'principled')).lower()
        if floor_material_kind not in {'principled', 'diffuse'}:
            logger.warning(
                f"Unknown enclosure.floor_material={floor_material_kind!r}; using principled."
            )
            floor_material_kind = 'principled'
        if floor_material_kind == 'diffuse':
            # Keep diffuse floors textureless, matching enclosure wall appearance.
            texture_path = None
            texture_scale = None

        size_x, size_y = float(size_xy[0]), float(size_xy[1])
        center_x, center_y = float(center_xy[0]), float(center_xy[1])
        if floor_material_kind == 'diffuse':
            floor_material = create_diffuse_material(
                f"FloorMaterial_{scene_name}", floor_color
            )
        else:
            floor_material = create_principled_material(
                f"FloorMaterial_{scene_name}", floor_color, floor_roughness, floor_metallic
            )
        floor_obj = add_box_primitive(
            f"{scene_name}_floor",
            (size_x, size_y, float(thickness)),
            (center_x, center_y, float(top_z) - float(thickness) / 2.0),
            floor_material,
            flip_normals=False,
        )
        floor_container = blender_utils.ObjContainer(
            [floor_obj], with_empty=False, recenter=False, rescale=False, file_name=source_name
        )
        if texture_path is not None:
            floor_container.apply_texture(texture_path, texture_scale if texture_scale is not None else 1.0)
        floor_meta = {
            'name': floor_obj.name,
            'kind': 'floor',
            'translation': [float(center_x), float(center_y), float(top_z) - float(thickness) / 2.0],
            'size': [size_x, size_y, float(thickness)],
        }
        if source_name is not None:
            floor_meta['source'] = source_name
        if texture_path is not None:
            floor_meta['texture'] = texture_path
            floor_meta['texture_scale'] = texture_scale
        return floor_container, floor_meta

    def add_enclosure(scene_name, floor_bounds=None, floor_top_z=None, source_name=None):
        enclosure_cfg = FLAGS.enclosure if FLAGS.enclosure is not None else {}
        if not enclosure_cfg.get('enabled', False):
            return []

        size_xy = enclosure_cfg.get('size')
        center_x = float((placement_vmin[0] + placement_vmax[0]) * 0.5)
        center_y = float((placement_vmin[1] + placement_vmax[1]) * 0.5)
        floor_z = float(placement_plane_offset[2])
        if floor_bounds is not None:
            floor_vmin = np.array(floor_bounds[0], dtype=np.float32)
            floor_vmax = np.array(floor_bounds[1], dtype=np.float32)
            center_x = float((floor_vmin[0] + floor_vmax[0]) * 0.5)
            center_y = float((floor_vmin[1] + floor_vmax[1]) * 0.5)
            floor_z = float(floor_vmax[2])
        if floor_top_z is not None:
            floor_z = float(floor_top_z)
        if size_xy is None:
            if floor_bounds is not None:
                size_xy = (floor_vmax - floor_vmin)[:2].tolist()
            else:
                size_xy = placement_range.tolist()
        size_x, size_y = float(size_xy[0]), float(size_xy[1])
        wall_height = float(enclosure_cfg.get('height', 2.0))
        wall_thickness = float(enclosure_cfg.get('thickness', 0.1))
        seam_epsilon = float(enclosure_cfg.get('seam_epsilon', 0.001))
        include_ceiling = bool(enclosure_cfg.get('ceiling', False))
        wall_color = enclosure_cfg.get('color', [0.85, 0.85, 0.85])
        wall_roughness = float(enclosure_cfg.get('roughness', 0.8))
        wall_metallic = float(enclosure_cfg.get('metallic', 0.0))

        if size_x <= 0 or size_y <= 0 or wall_height <= 0 or wall_thickness <= 0:
            raise ValueError(f"Invalid enclosure dimensions: {enclosure_cfg}")

        wall_material = create_principled_material(
            f"EnclosureMaterial_{scene_name}", wall_color, wall_roughness, wall_metallic
        )
        enclosure_meta = []
        wall_height_with_seam = wall_height + seam_epsilon
        wall_center_z = floor_z + wall_height * 0.5 - seam_epsilon * 0.5
        box_specs = [
            (
                'wall_south',
                (size_x, wall_thickness, wall_height_with_seam),
                (center_x, center_y - size_y / 2.0 - wall_thickness / 2.0, wall_center_z),
            ),
            (
                'wall_north',
                (size_x, wall_thickness, wall_height_with_seam),
                (center_x, center_y + size_y / 2.0 + wall_thickness / 2.0, wall_center_z),
            ),
            (
                'wall_west',
                (wall_thickness, size_y, wall_height_with_seam),
                (center_x - size_x / 2.0 - wall_thickness / 2.0, center_y, wall_center_z),
            ),
            (
                'wall_east',
                (wall_thickness, size_y, wall_height_with_seam),
                (center_x + size_x / 2.0 + wall_thickness / 2.0, center_y, wall_center_z),
            ),
        ]
        if include_ceiling:
            box_specs.append(
                (
                    'ceiling',
                    (size_x, size_y, wall_thickness),
                    (center_x, center_y, floor_z + wall_height - seam_epsilon + wall_thickness / 2.0),
                )
            )

        for part_name, size_xyz, location in box_specs:
            obj = add_box_primitive(
                f"{scene_name}_{part_name}", size_xyz, location, wall_material, flip_normals=True
            )
            part_meta = {
                'name': obj.name,
                'kind': part_name,
                'translation': [float(v) for v in location],
                'size': [float(v) for v in size_xyz],
                'base_color': [float(v) for v in wall_color],
                'roughness': wall_roughness,
                'metallic': wall_metallic,
            }
            if source_name is not None:
                part_meta['source'] = source_name
            enclosure_meta.append(part_meta)

        logger.info(
            f"Added enclosure for scene {scene_name}: size=({size_x}, {size_y}), "
            f"height={wall_height}, ceiling={include_ceiling}"
        )
        return enclosure_meta
    
    start_idx = 0
    iter_start_time = time.time()
    obj_iter = iter(obj_dataloader)

    def compose_scene(scene_name):
        placement_grid.fill(0)
        blender_utils.clear_scene()
        mesh_list = []
        mesh_meta = []
        floor_bounds = None
        floor_top_z = float(placement_plane_offset[2])
        floor_source_name = None

        # sample placement plane
        if num_planes > 0:
            plane_idx = np.random.choice(num_planes, size=1, p=FLAGS.plane_sample_weight)[0]
            placement_plane = blender_utils.add_object_file(
                placement_plane_path[plane_idx], with_empty=True, recenter=True, rescale=True
            )
            placement_plane.apply_transform((0, 0, 0), scale=placement_plane_scale)
            plane_vmin, plane_vmax = placement_plane.aabb
            vz = plane_vmin[2]
            vplane = np.array(placement_plane_offset) - np.array([0, 0, vz])
            placement_plane.apply_transform(vplane)

            plane_source_name = os.path.basename(placement_plane_path[plane_idx])
            floor_source_name = plane_source_name
            plane_meta = {'name': plane_source_name}
            plane_texture_path = None
            texture_scale = None
            if num_plane_textures > 0:
                plane_tex_idx = np.random.choice(
                    num_plane_textures, size=1, p=FLAGS.texture_sample_weight
                )[0]
                texture_scale = float(np.random.uniform(1.5, 2.5))
                plane_texture_path = plane_textures_path[plane_tex_idx]

            enclosure_cfg = FLAGS.enclosure if FLAGS.enclosure is not None else {}
            use_procedural_floor = enclosure_cfg.get('enabled', False) and enclosure_cfg.get('procedural_floor', True)
            if use_procedural_floor:
                plane_bounds = placement_plane.aabb
                plane_center = (np.array(plane_bounds[0][:2]) + np.array(plane_bounds[1][:2])) * 0.5
                floor_size_xy = enclosure_cfg.get('size')
                if floor_size_xy is None:
                    floor_size_xy = (np.array(plane_bounds[1][:2]) - np.array(plane_bounds[0][:2])).tolist()
                floor_thickness = float(enclosure_cfg.get('floor_thickness', enclosure_cfg.get('thickness', 0.1)))
                placement_plane.clear_objects()
                del placement_plane
                placement_plane, plane_meta = add_floor_slab(
                    scene_name,
                    floor_size_xy,
                    plane_center,
                    floor_top_z,
                    floor_thickness,
                    texture_path=plane_texture_path,
                    texture_scale=texture_scale,
                    source_name=plane_source_name,
                )
            else:
                if plane_texture_path is not None:
                    placement_plane.apply_texture(plane_texture_path, texture_scale)
                    plane_meta['texture'] = plane_texture_path
                    plane_meta['texture_scale'] = texture_scale

            floor_bounds = placement_plane.aabb
            floor_top_z = float(floor_bounds[1][2])
            mesh_meta.append(plane_meta)
            mesh_list.append(placement_plane)

        num_glbs = 0
        glb_attempts = 0
        while num_glbs < FLAGS.glbs_per_scene and glb_attempts < FLAGS.glbs_max_sample_tries_per_scene:
            target = sample_glb(obj_iter, obj_idx=num_glbs)
            glb_attempts += 1
            if target is None:
                continue
            ref_mesh, meta = target
            mesh_list.append(ref_mesh)
            mesh_meta.append(meta)
            num_glbs += 1

        if num_glbs < FLAGS.glbs_per_scene:
            logger.warning(
                f"Scene {scene_name}: placed {num_glbs}/{FLAGS.glbs_per_scene} GLBs after "
                f"{glb_attempts} samples"
            )
            return None, None

        shape_list = []
        shape_meta = []
        for j in range(FLAGS.shapes_per_scene):
            target = sample_shape(baseshape_files, obj_idx=j + num_glbs)
            if target is not None:
                ref_mesh, meta = target
                shape_list.append(ref_mesh)

                if FLAGS.sample_shape_texture and num_plane_textures > 0:
                    if random.random() < 0.25:
                        plane_tex_idx = np.random.choice(
                            num_plane_textures, size=1, p=FLAGS.texture_sample_weight
                        )[0]
                        texture_scale = 1.0
                        ref_mesh.apply_texture(plane_textures_path[plane_tex_idx], texture_scale)
                        meta['texture'] = plane_textures_path[plane_tex_idx]
                        meta['texture_scale'] = texture_scale
                shape_meta.append(meta)

        if len(shape_list) > 0:
            mesh_list.extend(shape_list)
            mesh_meta.extend(shape_meta)
        else:
            logger.info("No shapes added")

        enclosure_meta = add_enclosure(
            scene_name,
            floor_bounds=floor_bounds,
            floor_top_z=floor_top_z,
            source_name=floor_source_name,
        )
        if len(enclosure_meta) > 0:
            mesh_meta.extend(enclosure_meta)

        return mesh_list, mesh_meta

    for i in range(start_idx, FLAGS.num_rendering):
        logger.info(f"Rendering iteration {i}/{FLAGS.num_rendering}")
        name = f"{i:06d}"
        if FLAGS.dump_complete:
            new_complete_file = os.path.join(FLAGS.out_dir, f"COMPLETE_{name}")
            if os.path.exists(new_complete_file):
                logger.info(f"COMPLETE_{name} already exists, skip")
                continue
        prefix = ''
        if FLAGS.num_frames > 1:
            prefix = f"{0:04d}."
        mesh_list, mesh_meta = None, None
        for compose_try in range(FLAGS.scene_compose_retry_limit):
            mesh_list, mesh_meta = compose_scene(name)
            if mesh_list is not None:
                if compose_try > 0:
                    logger.info(
                        f"Scene {name} succeeded after {compose_try + 1} composition attempts"
                    )
                break
            logger.warning(
                f"Retrying scene {name} composition "
                f"({compose_try + 1}/{FLAGS.scene_compose_retry_limit})"
            )

        if mesh_list is None:
            raise RuntimeError(
                f"Failed to place {FLAGS.glbs_per_scene} GLBs in scene {name} after "
                f"{FLAGS.scene_compose_retry_limit} composition attempts"
            )

        # Render multiple views and dump results
        if not FLAGS.prefix_in_folder:
            os.makedirs(os.path.join(FLAGS.out_dir, name), exist_ok=True)

        if FLAGS.video_mode == 'drop_phy':
            from modes.drop_physics import run as run_drop_physics
            render_fn = run_drop_physics
            logger.warning("Drop physics is not fully tested")
        else:
            render_fn = render_scene

        render_fn(mesh_list, mesh_meta, envlight_path_list, name, prefix, FLAGS)

        post_process_rendering(
            os.path.join(FLAGS.out_dir, name),
            feature_fmt=FLAGS.dump_format,
            dump_video=FLAGS.dump_video
        )

        if FLAGS.dump_blend:
            # save blender scene
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FLAGS.out_dir, name, f"scene.blend"))
        if FLAGS.dump_placement:
            # Plot placement grid with some color map
            fig = plt.figure()
            plt.imshow(placement_grid.T, cmap='tab20', vmin=0, vmax=placement_grid.max().item() + 1)
            # plt.axis('off')
            plt.savefig(os.path.join(FLAGS.out_dir, name, f"placement.png"))
            plt.close(fig)
        if FLAGS.dump_complete:
            new_complete_file = os.path.join(FLAGS.out_dir, f"COMPLETE_{name}")
            open(new_complete_file, 'w').close()

    # The process is about to exit, so avoid one extra Blender teardown pass
    # here. Final clear_scene()/quit operations have been observed to segfault
    # after all outputs are already written.
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))
