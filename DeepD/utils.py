"""
This module defines helper functions for training
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1.errors import OutOfRangeError
import hashlib
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class TimeLogger:
    """
    Helper tool for logging training time
    """
    def __init__(self, time_logger_step=1, hierachy=1):
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s):
        """time log"""
        if self.step_count % self.time_logger_step == 0:
            print("[Utils] " + "#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))
            self.time = time.time()
            self.step_count += 1


class Screenshot(dict):
    """
    Helper tool for logging and monitoring training process
    """
    def __init__(self, model, n_iter_buffer, verbose=2, listen_freq=50):
        """
        :param model: (DeepD) model instance from ./model.py
        :param n_iter_buffer: (int) moving average window width for loss
        :param verbose: (int) verbose: 0: export model params; 1: export real time loss; 2: export model checkpoint;
                                       3: export encoded tp2vecs; 4: export reconstructed data matrix
        :param listen_freq: (int) the listening frequency (for stdout only), full loss would be saved in training.log
        """
        # initialize loss_min
        super().__init__()
        self.n_iter_buffer = n_iter_buffer
        self.model = model
        self.verbose = verbose
        self.freq = listen_freq
        self.reset()

    def avg_n_iters_loss(self, new_loss):
        self.saved_losses.append(new_loss)
        self.saved_losses = self.saved_losses[-self.n_iter_buffer:]
        self.buffer_loss = sum(self.saved_losses) / len(self.saved_losses)
        return self.buffer_loss

    def screenshot(self, loss_min):
        self.loss_min = loss_min
        model = self.model
        params = model.sess.run(model.export_params)
        for item in params:
            params[item] = pd.DataFrame(params[item])
        self.update(params)

    def save_output(self, output, tensor_name, require_verbose):
        try:
            os.mkdir("outputs")
        except FileExistsError:
            pass
        for out, name, v in zip(output, tensor_name, require_verbose):
            if self.verbose > v:
                pd.DataFrame(out).to_csv('outputs/' + name + '.csv', header=None, index=None)

    def check_exist_params(self):
        return glob.glob("best.*.csv") != []

    def save_params(self):
        try:
            os.mkdir("params")
        except FileExistsError:
            pass
        for file in glob.glob("params/best.*.csv"):
            os.remove(file)
        for key in self.model.export_params:
            self[key].to_csv("params/best.{}.loss.{}.csv".format(key, self.loss_min))

    def save_model(self, path):
        if self.verbose > 1:
            tmp = self.model.saver.save(self.model.sess, path)
            print("[Utils] Model saved in path: %s" % os.getcwd() + '/' + tmp)

    def load_model(self, path):
        tmp = self.model.saver.restore(self.model.sess, path)
        print("[Utils] Model restored from path: %s" % os.getcwd() + '/' + tmp)

    def log(self, filename, iteration, loss, unchanged, t):
        idx_iter, n_iter = iteration
        n_unchanged, n_iter_patience = unchanged
        loss_train_i, loss_valid_i, loss_train_mse_i, loss_valid_mse_i, loss_test_mse = loss

        if self.verbose > 0:
            log_text = "Iteration: {}/{}\t" \
                       "loss (train):{:1.6f}\tloss (buffer on valid):{:1.6f}\t" \
                       "best:{:1.6f}\tTolerance: {}/{}\tTime_elapsed: {}" \
                .format(idx_iter, n_iter, loss_train_i, self.buffer_loss, self.loss_min,
                        n_unchanged, n_iter_patience, t)
            if self.counter % self.freq == 0:
                print(log_text)

        if not os.path.exists(filename):
            with open(filename, 'a') as f:
                log_text = "iter, loss_train, loss_valid, mse_train, mse_valid, loss_buffer, loss_min, " \
                           "mse_test, n_unchanged, n_iter_patience, time_elapsed\n"
                f.write(log_text)

        with open(filename, 'a') as f:
            content = (idx_iter, loss_train_i, loss_valid_i, loss_train_mse_i, loss_valid_mse_i,
                       self.buffer_loss, self.loss_min, loss_test_mse, n_unchanged, n_iter_patience, t)
            log_text = ",".join([str(i) for i in content])
            f.write(log_text + '\n')

        self.counter += 1

    def reset(self):
        self.loss_min = 1000
        self.saved_losses = [self.loss_min]
        self.counter = 0


def md5(key):
    """
    Helper tool for providing unique md5 ids to exprs
    """
    return hashlib.md5(key.encode()).hexdigest()


def get_metrics(logits, gold, name="DeepDCancer"):
    yhat = np.argmax(logits, axis=1)
    gold = np.argmax(gold, axis=1)
    for i in ["confusion_matrix", "roc_auc_score", "balanced_accuracy_score", "accuracy_score"]:
        print("[Evaluation] ", i)
        print(eval(i)(gold, yhat))
        if i == 'confusion_matrix':
            plt.subplots(figsize=[4, 4])
            sns.heatmap(eval(i)(gold, yhat), cmap='viridis', annot=True, fmt="g")
            plt.title("Confusion Matrix".format(i))
            plt.savefig("outputs/{}.png".format(name))

    return yhat


def plot_reconstruction(xhat, x, zhat, y, n=100):
    try:
        os.mkdir("outputs")
    except FileExistsError:
        pass

    print("[Utils] Plotting Tp2vec reconstruction results...")
    plt_pos = np.random.choice(range(zhat.shape[0]), n)
    plt.subplots(figsize=[18, 6])

    plt.subplot(131)
    sns.kdeplot(x=x[plt_pos].flatten(), y=xhat[plt_pos].flatten(), shade=True, cmap='mako')
    plt.title("Reconstruction")
    plt.xlabel("Original genee expression")
    plt.ylabel("Reconstructed gene expression")

    plt.subplot(132)
    sns.histplot(x=x[plt_pos].flatten(), y=xhat[plt_pos].flatten(), bins=40, palette='viridis')
    plt.title("Reconstruction")
    plt.xlabel("Original genee expression")
    plt.ylabel("Reconstructed gene expression")

    y = np.argmax(y, axis=1)
    nclass = max(y) + 1
    for i in range(nclass):
        plt.subplot(1, 3 * nclass, 2 * nclass + i + 1)
        pos = np.where(y == i)
        sns.heatmap(zhat[pos], cmap='viridis')
        plt.title("Extracted features (class: {})".format(i))

    plt.tight_layout()
    plt.savefig("outputs/DeepT2Vec.png")
