import glob, multiprocessing as mp, torch, json, argparse
import torch, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/ test)', default='train')
opt = parser.parse_args()

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + 'ing_data/data/*_datas.pth'))
files2 = sorted(glob.glob(split + 'ing_data/data/*_syminfo.json'))
assert len(files) == len(files2)

def f_test(fn):
    fn2 = fn[:-9] + 'syminfo.json'
    image_pc, instance_label, center_trans, scales = torch.load(fn)
    with open(fn2, 'r') as f:
        geo_syms = json.load(f)
    geo_syms = geo_syms["geo_syms"]
    for target, label, sym, center_tran, scale in zip(image_pc, instance_label, geo_syms, center_trans, scales):
        fs = fn[:-9]+sym+'_'+str(int(label))+'.pth'
        torch.save([target, center_tran, scale], fs)
    print('Processed ' + fn)


def f(fn):
    fn2 = fn[:-9] + 'syminfo.json'
    image_pc, instance_label, gt_pose, center_trans, scales = torch.load(fn)
    print(fn2)
    try:
        with open(fn2, 'r') as f:
            geo_syms = json.load(f)
    except:
        print('ERROR', fn2)
        assert False
    geo_syms = geo_syms["geo_syms"]
    for target, label, sym, gtpose, center_tran, scale in zip(image_pc, instance_label, geo_syms, gt_pose, center_trans, scales):
        try:
            assert len(sym) > 0
        except:
            print('ERROR', fn)
            assert False
        fs = fn[:-9]+sym+'_'+str(int(label))+'.pth'
        torch.save((target, gtpose, center_tran, scale), fs)
    print('Processed ' + fn)

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
