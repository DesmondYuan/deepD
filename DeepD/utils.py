"""
This module defines the training of the model
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1.errors import OutOfRangeError
import hashlib
import pandas as pd


class TimeLogger:
    """calculate training time"""

    def __init__(self, time_logger_step=1, hierachy=1):
        self.time_logger_step = time_logger_step
        self.step_count = 0
        self.hierachy = hierachy
        self.time = time.time()

    def log(self, s):
        """time log"""
        if self.step_count % self.time_logger_step == 0:
            print("#" * 4 * self.hierachy, " ", s, "  --time elapsed: %.2f" % (time.time() - self.time))
            self.time = time.time()
            self.step_count += 1


class Screenshot(dict):
    def __init__(self, model, n_iter_buffer, verbose=2):
        # initialize loss_min
        super().__init__()
        self.n_iter_buffer = n_iter_buffer
        self.model = model
        self.verbose = verbose
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

    def save_output(self, X):
        z, xhat = self.model.sess.run([self.model.encoders[-1]['tensor'], self.model.decoders[-1]['tensor']],
                                      feed_dict={self.model.encoders[0]['tensor']: X})
        if self.verbose > 1:
            pd.DataFrame(z).to_csv("features.csv", header=None, index=None)
            pd.DataFrame(xhat).to_csv("imputation.csv", header=None, index=None)
        return z, xhat

    def check_exist_params(self):
        return glob.glob("best.*.csv") != []

    def save_params(self):
        for file in glob.glob("best.*.csv"):
            os.remove(file)
        for key in self.model.export_params:
            self[key].to_csv("best.{}.loss.{}.csv".format(key, self.loss_min))

    def save_model(self, path):
        if self.verbose > 2:
            tmp = self.model.saver.save(self.model.sess, path)
            print("Model saved in path: %s" % os.getcwd() + tmp[1:])

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

    def reset(self):
        self.loss_min = 1000
        self.saved_losses = [self.loss_min]


def md5(key):
    return hashlib.md5(key.encode()).hexdigest()
