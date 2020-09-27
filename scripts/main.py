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
from DeepD.train import pretrain, train, session_setup, session_init
from DeepD.data import random_partition


parser = argparse.ArgumentParser(description='DeepD main script')
parser.add_argument('-config', '--experiment_config_path', required=True, type=str, help="Path of experiment config")
parser.add_argument('-seed', '--random_seed', required=False, default=0, type=int, help="Random seed")
args = parser.parse_args()
cfg = json.load(open(args.experiment_config_path, 'r'))
print("[Main] Experiment starts with seed={} and config={}".format(args.random_seed, cfg))

# Loading datasets
print("[Main] Loading datasets...")
train_df = pd.read_csv(cfg['train_dataset'], index_col=0)
test_df = pd.read_csv(cfg['test_dataset'], index_col=0)
train_set, valid_set, test_set, valid_pos = random_partition(train_df=train_df, test_df=test_df, seed=args.random_seed,
                                                             annotation_col=cfg['annotation_col'],
                                                             validation_ratio=cfg['validation_ratio'])
datasets = {'train': train_set, 'valid': valid_set, 'test': test_set}
print("[Main] All datasets are ready...")

# Prepareing working dir
print("[Main] Initializing working directory...")
run_id = cfg['expr_name'] + '_' + md5(str(cfg))
wdr = 'results/{}/seed_{}'.format(run_id, args.random_seed)
shutil.rmtree(wdr, ignore_errors=True)
os.makedirs(wdr, exist_ok=True)
os.chdir(wdr)
pd.DataFrame(valid_pos).to_csv("valid_set.pos")
valid_set['full_label'].to_csv("valid_set.labels.csv")

# Constructing DeepD model
print("[Main] Starting model construction at {}.".format(wdr))
for key in datasets:
    assert datasets[key]['value'].shape[1] == cfg['layers'][0][0]
print("[Main] Constructing DeepD model...")
model = DeepD(cfg, seed=args.random_seed)
print("[Main] Training DeepD model...")
model.pretrain(datasets, n_iter=cfg['max_iteration_pretrain'], n_iter_patience=cfg['n_iter_patience_pretrain'])
z, xhat = model.train(datasets, n_iter=cfg['max_iteration'], n_iter_patience=cfg['n_iter_patience'])

# Plotting results
print("[Main] Plotting results...")
plt_pos = np.random.choice(range(z.shape[0]), 100)
plt.subplots(figsize=[18, 6])
if sns.__version__ >= "0.11":
    plt.subplot(131)
    sns.kdeplot(x=test_set['value'][plt_pos].flatten(), y=xhat[plt_pos].flatten(), shade=True)
    plt.subplot(132)
    sns.histplot(x=test_set['value'][plt_pos].flatten(), y=xhat[plt_pos].flatten(), bins=40)
else:
    plt.subplot(131)
    sns.kdeplot(data=test_set['value'][plt_pos].flatten(), data2=xhat[plt_pos].flatten(), shade=True)
plt.subplot(133)
sns.heatmap(z[plt_pos], cmap='viridis')
plt.tight_layout()
plt.savefig("results.png")
print('[Main] Experiment finishes.')
# Training DeepD models
print("[Main] Starting model training at {}.".format(wdr))
sess, saver = session_setup()
tp2vec.attach_sess(sess, saver)
deepdc.attach_sess(sess, saver)
session_init(sess=sess, seed=args.random_seed)

