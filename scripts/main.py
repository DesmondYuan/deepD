import os
import shutil
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DeepD.model import DeepD
from DeepD.utils import md5


parser = argparse.ArgumentParser(description='DeepD main script')
parser.add_argument('-config', '--experiment_config_path', required=True, type=str, help="Path of experiment config")
parser.add_argument('-seed', '--random_seed', required=False, default=0, type=int, help="Random seed")
args = parser.parse_args()
cfg = json.load(open(args.experiment_config_path, 'r'))

# Loading datasets
print("[Main] Loading datasets...")
train_set = pd.read_csv(cfg['train_dataset'], header=None).values[:, 1:]  # drop the annotation column 0
test_set = pd.read_csv(cfg['test_dataset'], header=None).values[:, 1:]
train_set, test_set = preprocess(train_set, test_set)

print("[Main] Creating validation set...")
np.random.seed(args.random_seed)
n_samples = train_set.shape[0]
pos = np.random.choice(range(n_samples), n_samples, replace=False)
train_pos, valid_pos = pos[:int(0.7 * n_samples)], pos[int(0.7 * n_samples):]
data, labels, classes = [{'train': train_set[key][train_pos], 'valid': train_set[key][valid_pos], 'test': test_set[key],
                         'export': test_set[key]} for key in ['value', 'full_label', 'class_annot']]
print("[Main] All datasets are ready...")

# Prepareing working dir
run_id = cfg['expr_name'] + '_' + md5(str(cfg))
wdr = 'results/{}/seed_{}'.format(run_id, args.random_seed)
shutil.rmtree(wdr, ignore_errors=True)
os.makedirs(wdr, exist_ok=True)
os.chdir(wdr)
pd.DataFrame(train_pos).to_csv("train_set.pos")
pd.DataFrame(valid_pos).to_csv("valid_set.pos")
print("[Main] Starting model construction at {}.".format(wdr))

# Constructing DeepD model
for key in data:
    assert data[key].shape[1] == cfg['layers'][0][0]
print("[Main] Constructing DeepD model...")
model = DeepD(cfg, seed=args.random_seed)
print("[Main] Training DeepD model...")
z, xhat = model.train(data, n_iter=cfg['max_iteration'], verbose=1)

# Plotting results
print("[Main] Plotting results...")
plt_pos = np.random.choice(range(z.shape[0]), 1000)
plt.subplots(figsize=[12, 6])
plt.subplot(121)
sns.kdeplot(data['export'][plt_pos].flatten(), xhat[plt_pos].flatten(), shade=True)
xlim = np.quantile(data['export'][plt_pos].flatten(), (0.05, 0.95))
plt.xlim(xlim)
plt.ylim(xlim)
plt.subplot(122)
sns.heatmap(z[plt_pos], cmap='viridis')
plt.tight_layout()
plt.savefig("results.png")
print('[Main] Experiment finishes.')

