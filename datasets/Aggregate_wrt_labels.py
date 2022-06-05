import glob, multiprocessing as mp, torch, json, argparse
import torch, pickle, os

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train/ test)', default='train')
opt = parser.parse_args()

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

split = opt.data_split
print('data split: {}'.format(split))
labels = [i for i in range(79)]

def f_test(label:int):
    files = sorted(glob.glob(split + 'ing_data/data/*_'+str(label)+'.pth'))
    image_pc = []
    syms = []
    names = []
    prefix = split + 'ing_data/data/'
    for fn in files:
        target, center_tran, scale = torch.load(fn)
        sym = fn.split('_')[-2]
        try:
            assert len(sym) > 0
        except:
            print('ERROR', fn)
            assert False
        scene_name = fn.split('/')[-1].split('_')[0]
        image_pc.append([target, center_tran, scale])
        syms.append(sym)
        names.append(scene_name)
        print('Processed ' + fn)

    fdata = prefix + 'aggregated/' + str(label) + '.pth'
    torch.save(image_pc, fdata)
    ftext = prefix + 'aggregated/' + str(label) + '.json'
    with open(ftext, 'w') as f:
        json.dump({'syms': syms, 'names': names}, f)
    print('Generated ' + fdata)

def f(label:int):
    files = sorted(glob.glob(split + 'ing_data/data/*_'+str(label)+'.pth'))
    image_pc = []
    syms = []
    names = []
    prefix = split + 'ing_data/data/'
    for fn in files:
        target, gt_pose, center_tran, scale = torch.load(fn)
        sym = fn.split('_')[-2]
        scene_name = fn.split('/')[-1].split('_')[0]
        image_pc.append([target, gt_pose, center_tran, scale])
        syms.append(sym)
        names.append(scene_name)
        print('Processed ' + fn)

    fdata = prefix + 'aggregated/' + str(label) + '.pth'
    torch.save(image_pc, fdata)
    ftext = prefix + 'aggregated/' + str(label) + '.json'
    with open(ftext, 'w') as f:
        json.dump({'syms': syms, 'names': names}, f)
    print('Generated ' + fdata)

p = mp.Pool(processes=mp.cpu_count()//2) # Use all CPUs available
if opt.data_split == 'test':
    p.map(f_test, labels)
else:
    p.map(f, labels)
p.close()
p.join()
