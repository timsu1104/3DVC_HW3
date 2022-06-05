import sys, os
import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())
from utils.data_processing import *

class Dataset:
    def __init__(self): 
        self.batch_size = 4
        self.train_workers = 4
        self.val_workers = 4
        self.test_workers = 4
    
    def trainLoader(self, logger, label, toy=False):
        fdata, ftext = get_aggregated_file('train', label)
        with open(ftext, 'r') as f:
            text = json.load(f)
            sym = text['syms']
        data = torch.load(fdata)
        assert len(data) == len(sym)
        if toy: data, sym = data[:100], sym[:100]
        self.train_files = [[i, j] for i, j in zip(data, sym)]

        self.NumTrainSamples = len(self.train_files)
        logger.info('Training samples: {}'.format(int(self.NumTrainSamples * 0.8)))
        assert int(self.NumTrainSamples) > 0

        train_set = list(range(int(self.NumTrainSamples * 0.8)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def valLoader(self, logger, label, toy=False):
        fdata, ftext = get_aggregated_file('val', label)
        with open(ftext, 'r') as f:
            text = json.load(f)
            sym = text['syms']
        data = torch.load(fdata)
        assert len(data) == len(sym)
        if toy: data, sym = data[:100], sym[:100]
        self.val_files = [[i, j] for i, j in zip(data, sym)]

        self.NumValSamples = len(self.val_files)
        logger.info('Validation samples: {}'.format(self.NumValSamples - int(self.NumValSamples * 0.8)))
        assert self.NumValSamples - int(self.NumValSamples * 0.8) > 0

        val_set = list(range(int(self.NumValSamples * 0.8), self.NumValSamples))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    
    
    def testLoader(self, logger, label):
        fdata, ftext = get_aggregated_file('test', label)
        with open(ftext, 'r') as f:
            text = json.load(f)
            sym = text['syms']
            name = text['names']
        data = torch.load(fdata)
        assert len(data) == len(sym)
        self.test_files = [[i, j, k] for i, j, k in zip(data, sym, name)]

        self.NumTestSamples = len(self.test_files)
        logger.info('Testing samples: {}'.format(self.NumTestSamples))
        assert self.NumTestSamples > 0

        test_set = list(range(self.NumTestSamples))
        self.test_data_loader = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.testMerge, num_workers=self.test_workers, shuffle=True, sampler=None, drop_last=True, pin_memory=True)    

    def trainMerge(self, id):
        geo_syms = []
        image_pc = []
        gt_poses = []
        center_trans = []
        scales = []
        Length = 0
        for idx in id:
            data, geo_sym = self.train_files[idx]
            target, gt_pose, center_tran, scale = data
            if target[:, :3].max() < 1e-6 and target[:, :3].min() > -1e-6:
                continue
            geo_syms.append(geo_sym)
            image_pc.append(target)
            if Length < target.shape[0]: Length = target.shape[0]
            gt_poses.append(gt_pose)
            center_trans.append(center_tran)
            scales.append(scale)

        image_pc = pad_sequence(image_pc, batch_first=True)# torch.tensor(image_pc, dtype=torch.float32)
        # for i, target in enumerate(image_pc):
        #     new_target = torch.nn.functional.interpolate(target.T.unsqueeze(0), size=Length, mode='linear', align_corners=True).squeeze().T
        #     image_pc[i] = new_target
        # image_pc = torch.stack(image_pc, 0)
        gt_poses = torch.stack(gt_poses, 0)
        center_trans = torch.stack(center_trans, 0)
        scales = torch.stack(scales, 0)

        return {
            "image_pc": image_pc, 
            'symmetry': geo_syms, 
            "poses_world": gt_poses,
            "center_trans": center_trans,
            "scales": scales
            }
     
    def valMerge(self, id):
        geo_syms = []
        image_pc = []
        gt_poses = []
        center_trans = []
        scales = []
        for idx in id:
            data, geo_sym = self.val_files[idx]
            target, gt_pose, center_tran, scale = data
            if target[:, :3].max() < 1e-6 and target[:, :3].min() > -1e-6:
                continue
            geo_syms.append(geo_sym)
            image_pc.append(target)
            gt_poses.append(gt_pose)
            center_trans.append(center_tran)
            scales.append(scale)

        image_pc = pad_sequence(image_pc, batch_first=True)# torch.tensor(image_pc, dtype=torch.float32)
        gt_poses = torch.stack(gt_poses, 0)
        center_trans = torch.stack(center_trans, 0)
        scales = torch.stack(scales, 0)

        return {
            "image_pc": image_pc, 
            'symmetry': geo_syms, 
            "poses_world": gt_poses,
            "center_trans": center_trans,
            "scales": scales
            }

    def testMerge(self, id):
        geo_syms = []
        image_pc = []
        center_trans = []
        scales = []
        scene_name = []
        for idx in id:
            data, geo_sym, name = self.test_files[idx]
            target, center_tran, scale = data
            if target[:, :3].max() < 1e-6 and target[:, :3].min() > -1e-6:
                continue
            geo_syms.append(geo_sym)
            scene_name.append(name)
            image_pc.append(target)
            center_trans.append(center_tran)
            scales.append(scale)

        image_pc = pad_sequence(image_pc, batch_first=True)# torch.tensor(image_pc, dtype=torch.float32)
        center_trans = torch.stack(center_trans, 0)
        scales = torch.stack(scales, 0)

        return {
            "image_pc": image_pc, 
            'symmetry': geo_syms, 
            "center_trans": center_trans,
            "scales": scales, 
            "scene_name": scene_name
            }