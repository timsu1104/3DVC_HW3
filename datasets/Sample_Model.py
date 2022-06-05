import trimesh
import open3d as o3d
import numpy as np
import os, glob
from tqdm import tqdm, trange
import json

def read_models():
    MODELS = {}
    Paths = sorted(glob.glob('datasets/models/*/visual_meshes/'))
    for path in tqdm(Paths):
        with open(os.path.join(path, 'Samples.json'), 'r') as f:
            pc = json.load(f)
        name = 'models/' + path[:-1].split('/')[-2]
        MODELS[name] = pc
    return MODELS

if __name__ == '__main__':

    Paths = sorted(glob.glob('datasets/models/*/visual_meshes/'))
    for path in tqdm(Paths):
        dae_path = path + 'visual.dae'
        resolver = trimesh.resolvers.FilePathResolver(path)
        params = trimesh.exchange.dae.load_collada(dae_path, resolver=resolver)
        assert len(params['geometry'].keys()) == 1
        for v in params['geometry'].values():
            mesh = trimesh.Trimesh(
                vertices=v['vertices'], 
                faces=v['faces'], 
                vertex_normals=v['vertex_normals'], 
                vertex_colors=v['vertex_colors'], 
                visual=v['visual']
            )
        sample = {}
        Sample, _ = trimesh.sample.sample_surface(mesh, 10000)
        points = o3d.utility.Vector3dVector(Sample)
        source = o3d.geometry.PointCloud()
        source.points = points
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.002, max_nn=30))
        sample['coords'] = Sample.tolist()
        sample['normals'] = np.array(source.normals).tolist()
        with open(os.path.join(path, 'Samples_pointnet.json'), 'w') as f:
            json.dump(sample, f)

    MODELS = read_models()
    with open('datasets/models_pointnet.json', 'w') as f:
        json.dump(MODELS, f)
    """
    python datasets/Sample_Model.py
    """