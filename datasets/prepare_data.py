'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, numpy as np, multiprocessing as mp, torch, json, argparse
import torch
from PIL import Image
import pickle
import pandas as pd

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/ test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + 'ing_data/data/*_color_kinect.png'))
files2 = sorted(glob.glob(split + 'ing_data/data/*_depth_kinect.png'))
files3 = sorted(glob.glob(split + 'ing_data/data/*_label_kinect.png'))
files4 = sorted(glob.glob(split + 'ing_data/data/*_meta.pkl'))
# print(len(files), len(files2))
assert len(files) == len(files2)
assert len(files) == len(files3)

train_df = pd.read_csv('training_data/objects_v1.csv')
test_df = pd.read_csv('testing_data/objects_v1.csv')

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

def f_test(fn):
    fn2 = fn[:-16] + 'depth_kinect.png'
    fn3 = fn[:-16] + 'label_kinect.png'
    fn4 = fn[:-16] + 'meta.pkl'
    rgb = np.array(Image.open(fn)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(fn2)) / 1000   # convert from mm to m
    label = np.array(Image.open(fn3))
    meta = load_pickle(fn4)
    coords, _ = export_one_scan(depth, meta, 'test')

    selected = test_df.loc[test_df['object'].isin(meta['object_names'])]
    symmetry = selected['geometric_symmetry'].to_list()
    ind = selected.index.to_numpy()

    geo_syms = []
    image_pc = []

    cnt = -1
    sel_ind = []
    center_trans = []
    scales = []
    for index in ind:
        cnt += 1
        scale = np.array(meta['scales'][index])

        # Read image target pose
        image_points = coords[label == index].reshape((-1,3))
        selected_colors = rgb[label == index].reshape((-1,3))
        if image_points.shape[0] == 0:
            continue
        target = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
        center_tran = target.mean(axis=0)
        target -= target.mean(axis=0) # zero centering
        target /= scale
        target = np.concatenate([target, selected_colors], axis=1)

        image_pc.append(torch.from_numpy(target).float())
        sel_ind.append(cnt)
        center_trans.append(center_tran)
        scales.append(scale)

    geo_syms = [symmetry[i] for i in sel_ind]
    instance_label = torch.tensor(ind[sel_ind], dtype=torch.float32)
    center_trans = torch.tensor(center_trans, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)

    torch.save((image_pc, instance_label, center_trans, scales), fn[:-16]+'datas.pth')
    with open(fn[:-16]+'syminfo.json', 'w') as f:
        json.dump({"geo_syms": geo_syms}, f)
    print('Saving to ' + fn[:-16]+'datas.pth')


def f(fn):
    fn2 = fn[:-16] + 'depth_kinect.png'
    fn3 = fn[:-16] + 'label_kinect.png'
    fn4 = fn[:-16] + 'meta.pkl'
    print(fn)

    rgb = np.array(Image.open(fn)) / 255   # convert 0-255 to 0-1
    depth = np.array(Image.open(fn2)) / 1000   # convert from mm to m
    label = np.array(Image.open(fn3))
    meta = load_pickle(fn4)
    coords, gt_pose, _ = export_one_scan(depth, meta)

    selected = train_df.loc[train_df['object'].isin(meta['object_names'])]
    symmetry = selected['geometric_symmetry'].to_list()
    ind = selected.index.to_numpy()

    geo_syms = []
    image_pc = []
    center_trans = []
    scales = []

    cnt = -1
    sel_ind = []
    for index in ind:
        cnt += 1
        scale = np.array(meta['scales'][index])

        # Read image target pose
        image_points = coords[label == index].reshape((-1,3))
        selected_colors = rgb[label == index].reshape((-1,3))
        if image_points.shape[0] == 0:
            continue
        target = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
        center_tran = target.mean(axis=0)
        target -= target.mean(axis=0) # zero centering
        target /= scale
        target = np.concatenate([target, selected_colors], axis=1)

        image_pc.append(torch.from_numpy(target).float())
        sel_ind.append(cnt)
        center_trans.append(center_tran)
        scales.append(scale)

    geo_syms = [symmetry[i] for i in sel_ind]
    gt_pose = torch.tensor(gt_pose[sel_ind], dtype=torch.float32)
    instance_label = torch.tensor(ind[sel_ind], dtype=torch.float32)
    center_trans = torch.tensor(center_trans, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    assert len(instance_label) == len(gt_pose) == len(image_pc) == len(geo_syms)

    torch.save((image_pc, instance_label, gt_pose, center_trans, scales), fn[:-16]+'datas.pth')
    assert len(geo_syms) > 0
    with open(fn[:-16]+'syminfo.json', 'w') as f:
        json.dump({"geo_syms": geo_syms}, f)
    print('Saving to ' + fn[:-16]+'datas.pth')

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
