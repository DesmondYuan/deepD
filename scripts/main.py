"""
This is the main script of DeepD v0.1
Developed and maintained by Xu lab at https://github.com/DesmondYuan/deepD
For quick start, try this with python scripts/main.py -config=configs/example_GENT_NC.json

The config file should include the following information
    "expr_name": (str) Label used as folder name under results
    "train_dataset": (str) Location of the dataset that would be further split into training set and validation set with
                     the "validation_ratio.
    "test_dataset": (str) Location of the withheld/test dataset.
    "annotation_col": (str) On which column of the input data frame would a supervised model be trained to classify.
    "validation_ratio": (float) The training/validation ratio for data partition used for "train_dataset".
    "n_genes": (int) Number of genes from the input data.
    "unsupervised_layers": (list) A list of layer sizes used for encoders and decoders.
    "supervised_layers": (list) A list of layer sizes used for supervised classifier DeepDCancer.
    "pretrain_tp2vec": (bool) Whether to perform unsupervised pretraining.
    "plot_pretrain_results": (bool) Whether to plot the results after pretraining.
    "train_disconnected_classifier": (bool) Whether to perform the disconnected supervised classification (DeepDCancer).
    "train_connected_classifier": (bool) Whether to perform the connected supervised classification (DeepDcCancer).
    "max_iteration": (int) Maximum number of iterations used for training.
    "max_iteration_pretrain": (int) Maximum number of iterations used for pretraining.
    n_iter_buffer (int): The moving window for eval losses during training.
    n_iter_patience (int): How many iterations without buffered loss on validation dataset decreases would result in
                           an earlystop in training.
    "n_iter_patience_pretrain":How many iterations without buffered loss on validation dataset decreases would result in
                           an earlystop in pretraining (for each layer).
    learning_rate (float): The learning rate for Adam Optimizer. Note that we used the default beta1 and beta2 for Adam.
    l1 (float): l1 regularization strength.
    l2 (float): l2 regularization strength.
    "activation": (tensorflow) Activation function for dense layers.
    "optimizer": (tensorflow) Which optimizer would be used for training.
    "verbose": (int) export verbose
    "listen_freq": (int) Printing training loss for each # of iterations.
    "pretrain_batch_size": Batch size for each iteration in pretraining.
    "batch_size": Batch size for each iteration in training.
"""
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
                    output=[tp2vec.encoders[-1]['tensor'], tp2vec.full_decoder['tensor']],
                    n_iter=cfg['max_iteration'], n_iter_patience=cfg['n_iter_patience'])
    tp2vec.screenshot.save_output([z, xhat], ["Compressed_features", "Reconstruction"], require_verbose=[2, 3])
    if cfg['plot_pretrain_results']:
        plot_reconstruction(xhat=xhat, x=test_set['value'], zhat=z, y=test_set['class_annot'], n=20)

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
