import glob
import os
import shutil
import random

DATA_DIR = '/mnt2/shaobo/evimo/sunny_record'

def num_data(data):
    samples = glob.glob(os.path.join(data, "processed", "*"))
    length = len(samples)
    return length

def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_val(data_dir, val_ratio):
    samples = glob.glob(os.path.join(data_dir, "processed_", "*"))
    n = len(samples)
    n_val = n * val_ratio
    val_cnt = 0
    p = 0.5
    train_dir = os.path.join(data_dir, 'processed', 'training')
    val_dir = os.path.join(data_dir, 'processed', 'validation')
    os.makedirs(os.path.dirname(train_dir), exist_ok=True)
    os.makedirs(os.path.dirname(val_dir), exist_ok=True)
    for sample in samples:
        rd = random.random()
        if rd >= p and val_cnt < n_val:
            copyfile(sample, val_dir)
            val_cnt += 1
        else:
            copyfile(sample, train_dir)
    

if __name__ == '__main__':
     reorg_train_val(DATA_DIR, 0.4)
