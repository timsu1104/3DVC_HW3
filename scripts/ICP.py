import sys, os, pandas as pd, argparse
from tqdm.contrib import tzip
import open3d as o3d
import json

sys.path.append(os.getcwd())
from utils.data_processing_ICP import *
from utils.eval import eval

NUM_OBJECTS = 79
with open('datasets/models.json', 'r') as f:
    MODELS = json.load(f)

def LoadAllData():
    rgb_files_train, depth_files_train, label_files_train, meta_files_train = get_split_files('train')
    rgb_files_val, depth_files_val, label_files_val, meta_files_val = get_split_files('val')
    rgb_files, depth_files, label_files, meta_files = \
        rgb_files_train + rgb_files_val, depth_files_train + depth_files_val,\
        label_files_train + label_files_val, meta_files_train + meta_files_val
    return rgb_files, depth_files, label_files, meta_files

def export_pc(df, rgb_file, depth_file, label_file, meta_file, voxel_size=0.001, task='train'):
    model_pc = []
    tgtpc = []
    trans_inits = []
    geo_syms = []
    gt_poses = []
    rgb, depth, label, meta = read_file(rgb_file, depth_file, label_file, meta_file)
    if task == 'train': 
        coords, gtpose, size = export_one_scan(depth, meta, task)
    else: 
        coords, size = export_one_scan(depth, meta, task)
    selected = df.loc[df['object'].isin(meta['object_names'])]
    symmetry = selected['geometric_symmetry'].to_list()
    loc = selected['location'].to_list()
    ind = selected.index.to_list()
    voxel_size = size.min()/20

    cnt = -1
    for directory, index in zip(loc, ind):
        cnt += 1
        scale = meta['scales'][index]
        
        # Read Ground Truth Poses
        sample = scale * MODELS[directory]['coords']
        # normals = scale * MODELS[directory]['normals']
        points = o3d.utility.Vector3dVector(sample)
        # normal = o3d.utility.Vector3dVector(normals)
        source = o3d.geometry.PointCloud()
        source.points = points
        # source.normals = normal
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        source.transform(np.identity(4, dtype=np.float64))

        # Read image target pose
        image_points = coords[label == index].reshape((-1,3))
        if image_points.shape[0] == 0:
            continue
        image_points = image_points @ np.linalg.inv(meta['extrinsic'])[:3, :3].T + np.linalg.inv(meta['extrinsic'])[:3, 3]
        points = o3d.utility.Vector3dVector(image_points)
        target = o3d.geometry.PointCloud()
        target.points = points
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        # Get transform initial alignment
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        trans_init = np.asarray(result.transformation)

        model_pc.append(source)
        tgtpc.append(target)
        trans_inits.append(trans_init)
        geo_syms.append(symmetry[cnt])
        if task == 'train':  gt_poses.append(gtpose[cnt])

    if task == 'train': 
        return gt_poses, geo_syms, model_pc, tgtpc, trans_inits, ind
    return geo_syms, model_pc, tgtpc, trans_inits, ind

def ICP(source, target, threshold=0.005, trans_init=np.identity(4), type='point'):
    if type == 'plane':
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))
    elif type == 'point':
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000))
    return reg_p2p.transformation

def solve(TASK, rgb_file, depth_file, label_file, meta_file, df, type):

    Total = 0
    Shots = 0
    ##### dataset
    if TASK == 'train':
        gt_pose, geo_syms, model_pc, tgtpc, trans_inits, label = export_pc(df, rgb_file, depth_file, label_file, meta_file, task=TASK)
        Total += len(model_pc)
    else: 
        geo_syms, model_pc, tgtpc, trans_inits, label = export_pc(df, rgb_file, depth_file, label_file, meta_file, task=TASK)
        pred_of_scene = ['null' for _ in range(NUM_OBJECTS)]

    assert len(model_pc) == len(tgtpc) == len(geo_syms) == len(trans_inits)
    for object_id in range(len(model_pc)):
        source = model_pc[object_id]
        target = tgtpc[object_id]
        symmetry = geo_syms[object_id]
        trans_init = trans_inits[object_id]
        pred = ICP(source, target, trans_init=trans_init, type=type)

        if TASK == 'train': 
            gt = gt_pose[object_id]
            r_diff, t_diff = eval(pred, gt, symmetry)
            if r_diff < 5 and t_diff < 1:
                Shots += 1
        else:
            pred_of_scene[label[object_id]] = pred.tolist()

    if TASK == 'test':
        scene_name = rgb_file.split('/')[-1].split('_')[0]
        global preds
        preds[scene_name] = {"poses_world": pred_of_scene}
    else:
        return Total, Shots

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test', default='train')
    parser.add_argument('--tag', help='', default='')
    parser.add_argument('--clip_l', help='', default='0')
    parser.add_argument('--clip', help='', default='1500')
    parser.add_argument('--type', help='point/plane', default='point')
    opt = parser.parse_args()
    TASK = opt.task
    TAG = opt.tag
    start = int(opt.clip_l)
    CLIP = int(opt.clip)
    type = opt.type
    assert TASK == 'train' or TASK == 'test'
    df = pd.read_csv('datasets/' + TASK + 'ing_data/objects_v1.csv')

    rgb_files, depth_files, label_files, meta_files = LoadAllData() if TASK == 'train' else get_split_files('test')

    if TASK == 'train':
        rgb_files, depth_files, label_files, meta_files = rgb_files[start:CLIP], depth_files[start:CLIP], label_files[start:CLIP], meta_files[start:CLIP]
        Total = 0
        Shots = 0
    else:
        global preds
        preds = {}

    for rgb_file, depth_file, label_file, meta_file in tzip(rgb_files, depth_files, label_files, meta_files):
        if TASK == 'train':
            Tadd, sadd = solve(TASK, rgb_file, depth_file, label_file, meta_file, df, type)
            Total += Tadd
            Shots += sadd
        else: solve(TASK, rgb_file, depth_file, label_file, meta_file, df, type)

    if TASK == 'train':
        acc = Shots * 1. / Total
        print("Accuracy is", acc)
    else:
        with open('outputs/ICP_'+ TAG + '.json', 'w') as f:
            json.dump(preds, f)

"""
points clip 3000: 0.699466310873916
"""