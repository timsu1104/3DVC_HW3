import numpy as np
import os, glob
from PIL import Image
import pickle
import open3d as o3d

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Change these 
training_data_dir = "datasets/training_data/data"
testing_data_dir = "datasets/testing_data/data"
split_dir = "datasets/training_data/splits"

def get_aggregated_file(split, label):
    if split == 'test':
        fdata = testing_data_dir + '/aggregated/' + label + '.pth'
        ftext = testing_data_dir + '/aggregated/' + label + '.json'
    else:
        fdata = training_data_dir + '/aggregated/' + label + '.pth'
        ftext = training_data_dir + '/aggregated/' + label + '.json'
    return fdata, ftext

def get_split_files(split_name):
    if split_name == 'test':
        data = sorted(glob.glob(os.path.join(testing_data_dir, '*_datas.pth')))
        syminfo = sorted(glob.glob(os.path.join(testing_data_dir, '*_syminfo.json')))
        return data, syminfo

    with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
        prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
        data = [p + "_datas.pth" for p in prefix]
        meta = [p + "_syminfo.json" for p in prefix]
    return data, meta

def read_file(rgb_file, depth_file, label_file, meta_file):
    rgb = np.array(Image.open(rgb_file)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(depth_file)) / 1000   # convert from mm to m
    label = np.array(Image.open(label_file))
    meta = load_pickle(meta_file)

    return rgb, depth, label, meta

def export_one_scan(depth, meta, task='train'):
    if task == 'train': 
        poses_world = np.array([meta['poses_world'][idx] for idx in meta['object_ids']])
    box_sizes = np.array([meta['extents'][idx] * meta['scales'][idx] for idx in meta['object_ids']])

    intrinsic = meta['intrinsic']
    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    coords = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    if task == 'train': 
        return coords, poses_world, box_sizes
    return coords, box_sizes

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result