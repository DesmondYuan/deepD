import sys
import os
import shutil
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.dirname(__file__) + '/..')
from DeepD.model import DeepT2vec, DeepDCancer
from DeepD.utils import md5, get_metrics, plot_reconstruction
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
json.dump(cfg, open('../config.json', 'w'), indent=4)

# Constructing DeepD models
for key in datasets:
    assert datasets[key]['value'].shape[1] == cfg['unsupervised_layers'][0][0]
print("[Main] Constructing DeepT2vec model at {}.".format(wdr))
tp2vec = DeepT2vec(cfg)
print("[Main] Constructing DeepDCancer and DeepDcCancer model at {}.".format(wdr))
deepdc = DeepDCancer(tp2vec, cfg)

# Training DeepD models
print("[Main] Starting model training at {}.".format(wdr))
sess, saver = session_setup()
tp2vec.attach_sess(sess, saver)
deepdc.attach_sess(sess, saver)
session_init(sess=sess, seed=args.random_seed)

if cfg['pretrain_tp2vec']:
    print("-"*80, '\n')
    print("[Main] Training DeepT2vec model...")
    pretrain(model=tp2vec, data=datasets,
             n_iter=cfg['max_iteration_pretrain'], n_iter_patience=cfg['n_iter_patience_pretrain'])
    z, xhat = train(model=tp2vec, optimizer_op=tp2vec.optimizer_op, data=datasets,
                    raw_loss=tp2vec.mse, full_loss=tp2vec.loss, model_name="Tp2vec_finetune",
                    output=[tp2vec.decoders[-1]['tensor'], tp2vec.full_decoder['tensor']],
                    n_iter=cfg['max_iteration'], n_iter_patience=cfg['n_iter_patience'])
    tp2vec.screenshot.save_output([z, xhat], ["Compressed_features", "Reconstruction"], require_verbose=[2, 3])
    # plot_reconstruction(xhat=xhat, x=test_set['value'], zhat=z, y=test_set['class_annot'], n=2000)

if cfg['train_disconnected_classifier']:
    print("-"*80, '\n')
    print("[Main] Training DeepDCancer model...")
    logits = train(model=deepdc, optimizer_op=deepdc.optimizer_op_disconnected, data=datasets,
                   raw_loss=deepdc.xent_loss, full_loss=deepdc.loss, output=[deepdc.yhat], model_name="DeepDCancer",
                   n_iter=cfg['max_iteration'], n_iter_patience=cfg['n_iter_patience'])[0]
    yhat = get_metrics(logits, datasets['test']['class_annot'], name="DeepDCancer")
    deepdc.screenshot.save_output([logits, yhat], require_verbose=[1, 0],
                                  tensor_name=["Prediction_DeepDCancer_logits", "Prediction_DeepDCancer_class"])

if cfg['train_connected_classifier']:
    print("-"*80, '\n')
    print("[Main] Training DeepDcCancer model...")
    logits = train(model=deepdc, optimizer_op=deepdc.optimizer_op_connected, data=datasets,
                   raw_loss=deepdc.xent_loss, full_loss=deepdc.loss, output=[deepdc.yhat], model_name="DeepDcCancer",
                   n_iter=cfg['max_iteration'], n_iter_patience=cfg['n_iter_patience'])[0]
    yhat = get_metrics(logits, datasets['test']['class_annot'], name="DeepDcCancer")
    deepdc.screenshot.save_output([logits, yhat], require_verbose=[1, 0],
                                  tensor_name=["Prediction_DeepDcCancer_logits", "Prediction_DeepDcCancer_class"])

print('[Main] Experiment finishes.')
