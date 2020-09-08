import tensorflow as tf
import numpy as np
from . import utils
import time


class DeepD:

    def __init__(self, config, seed):
        self.config = config['layers']
        self.x = tf.compat.v1.placeholder("float", [None, self.config[0][0]], name='X')
        self.aF = eval(config['activation'])
        self.lr = config['learning_rate']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.batch_size = config['batch_size']
        self.reg_params, self.export_params = [], {}
        self.encoders = self.build_encoders(self.x)
        self.decoders = self.build_decoders(self.encoders)
        self.mse, self.loss = self.get_loss(self.x, self.decoders[-1]['tensor'])
        self.optimizer_op = get_optimizer(self.loss, self.lr)
        for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
            print(i)
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        np.random.seed(seed), tf.compat.v1.set_random_seed(seed)

    def construct_ae_layer(self, x, d_out=None, activationF=tf.tanh, weight=None, regularize=True, export=None):

        assert weight is not None or d_out is not None
        xaiver = tf.glorot_normal_initializer(seed=None)
        d_in = x.shape[1]
        if weight is None:
            weight = xaiver(shape=(d_in, d_out), dtype=tf.dtypes.float32)
        else:
            d_out = weight.shape[0]
            weight = tf.transpose(weight)
        bias = tf.Variable(tf.zeros(d_out))
        params = {'w': weight, 'b': bias}
        if regularize:
            self.reg_params.append(params['w'])
            self.reg_params.append(params['b'])
        if export is not None:
            self.export_params.update({'w_{}'.format(export): params['w']})
            self.export_params.update({'b_{}'.format(export): params['b']})
        y = activationF(tf.matmul(x, weight) + bias)
        return {'tensor': y, 'params': params}

    def build_encoders(self, x):
        config = self.config
        encoders = [{'tensor': x, 'params': {}}]
        for k, (p, q) in enumerate(config):
            x_in = encoders[-1]
            assert x_in['tensor'].shape[1] == p
            layer = self.construct_ae_layer(x=x_in['tensor'], d_out=q, activationF=self.aF,
                                            regularize=True, export='layer_{}'.format(k))
            encoders.append(layer)
        return encoders

    def build_decoders(self, encoders):
        decoders = []
        for encoder in encoders[:0:-1]:  #
            layer = self.construct_ae_layer(x=encoder['tensor'], weight=encoder['params']['w'], activationF=self.aF,
                                            regularize=False, export=None)
            decoders.append(layer)
        return decoders

    def get_loss(self, x, xhat):
        n_params = sum([np.prod(p.shape) for p in self.reg_params]).value
        l1 = self.l1 * tf.reduce_sum([tf.reduce_sum(tf.abs(p)) for p in self.reg_params]) / n_params
        l2 = self.l2 * tf.reduce_sum([tf.reduce_sum(tf.square(p)) for p in self.reg_params]) / n_params
        mse = tf.reduce_mean(tf.square(x - xhat))
        loss = mse + self.l1 + l2
        return mse, loss

    def train(self, data, n_iter_buffer=5, n_iter=1000, n_iter_patience=100, verbose=1):
        """
        :param data: (dict) training data dictionary
        :param n_iter_buffer: (int) moving average window width for loss
        :param n_iter: (int) maximum number of iterations allowed
        :param n_iter_patience: (int) tolerence of training loss no-decrease
        :param verbose: (int) verbose: 0: no output; 1: print real time loss; 2: export outputs; 3: export params
        :return: z, xhat: (numpy.array, numpy.array) feature matrix and imputation matrix
        """

        model, sess = self, self.sess
        screenshot = utils.Screenshot(self, n_iter_buffer, verbose=verbose)
        n_unchanged = 0
        idx_iter = 0
        x_train_gold, x_valid_gold, x_test_gold = (data[key] for key in ['train', 'valid', 'test'])
        # Training on train set batches with early stopping on valid set batched
        print('[Training] Training on train set...')
        while True:
            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break
            t0 = time.clock()
            pos_train = np.random.choice(range(x_train_gold.shape[0]), self.batch_size)
            pos_valid = np.random.choice(range(x_valid_gold.shape[0]), self.batch_size)
            _, loss_train_i, mse_train_i = sess.run((model.optimizer_op, model.loss, model.mse),
                                                    feed_dict={self.x: x_train_gold[pos_train]})

            # record training
            loss_valid_i, mse_valid_i = sess.run((model.loss, model.mse), feed_dict={self.x: x_valid_gold[pos_valid]})
            new_loss = screenshot.avg_n_iters_loss(loss_valid_i)
            screenshot.log(filename="training.log", iteration=(idx_iter, n_iter),
                           unchanged=(n_unchanged, n_iter_patience), t=time.clock() - t0,
                           loss=(loss_train_i, loss_valid_i, mse_train_i, mse_valid_i, np.nan))

            # early stopping
            idx_iter += 1
            if new_loss < screenshot.loss_min:
                n_unchanged = 0
                screenshot.screenshot(loss_min=new_loss)
            else:
                n_unchanged += 1

        # Evaluation on entire valid set
        print('[Training] Evaluating on valid set... {}'.format(x_valid_gold.shape))
        t0 = time.clock()
        loss_valid_i, mse_valid_i = sess.run((model.loss, model.mse), feed_dict={self.x: x_valid_gold})
        screenshot.log(filename="training.log", iteration=(-1, -1),
                       unchanged=(-1, -1), t=time.clock() - t0,
                       loss=(np.nan, loss_valid_i, np.nan, mse_valid_i, np.nan))

        # Evaluation on test set
        print('[Training] Evaluating on test set... {}'.format(x_test_gold.shape))
        t0 = time.clock()
        mse_test = sess.run(model.mse, feed_dict={self.x: x_test_gold})
        screenshot.log(filename="training.log", iteration=(-1, -1),
                       unchanged=(-1, -1), t=time.clock() - t0,
                       loss=(np.nan, np.nan, np.nan, np.nan, mse_test))

        # Save model
        screenshot.save_params()
        screenshot.save_model('./model.ckpt')
        z, xhat = screenshot.save_output(data['export'])
        return z, xhat


def get_optimizer(loss_in, lr, optimizer=tf.compat.v1.train.AdamOptimizer, var_list=None):
    if var_list is None:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    with tf.compat.v1.variable_scope("optimization", reuse=tf.compat.v1.AUTO_REUSE):
        opt = optimizer(lr)
        opt_op = opt.minimize(loss_in, var_list=var_list)
    return opt_op





