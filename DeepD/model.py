"""
This module defines main model modules
"""
import tensorflow as tf
import numpy as np
from . import utils
import time


class DeepD:
    """
    The main unsupervised DeepD model framework
    """
    def __init__(self, config, seed):
        """
        :param config: (dict) training configuration
        :param seed: (int) training random seed for both data partition (numpy), params initialization (numpy)
        """
        self.config = config['layers']
        self.x = tf.compat.v1.placeholder("float", [None, self.config[0][0]], name='X')
        self.aF = eval(config['activation'])
        self.lr = config['learning_rate']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.pretrain_batch_size = config['pretrain_batch_size']
        self.batch_size = config['batch_size']
        self.reg_params, self.export_params = [], {}
        self.encoders = self.build_encoders(self.x)
        self.decoders = self.build_pretrain_decoders(self.encoders)
        self.full_decoder = self.build_full_decoder(self.encoders)

        with tf.compat.v1.variable_scope("CDAE_Loss", reuse=tf.compat.v1.AUTO_REUSE):
            self.mse, self.loss = self.get_loss(self.x, self.full_decoder['tensor'])

        with tf.compat.v1.variable_scope("AEs_Loss", reuse=tf.compat.v1.AUTO_REUSE):
            self.pretrain_mses = [self.get_mse(x['tensor'], xhat['tensor'])
                                  for x, xhat in zip(self.encoders[:-1], self.decoders)]

        self.optimizer_op = get_optimizer(self.loss, self.lr, scope="CDAE_optimization", optimizer=config['optimizer'])
        self.pretrain_optimizer_ops = [
            get_optimizer(mse, self.lr, scope="pretrain_opt_{}".format(k+1), optimizer=config['optimizer'],
                          var_list=tf.compat.v1.get_collection('variables', scope='Encoder_{}'.format(k+1)))
            for k, mse in enumerate(self.pretrain_mses)]
        self.screenshot = utils.Screenshot(self, config['n_iter_buffer'], verbose=config['verbose'],
                                           listen_freq=config['listen_freq'])

    def attach_sess(self, sess, saver):
        self.sess = sess
        self.saver = saver

    def construct_ae_layer(self, x, d_out=None, activationF=tf.tanh, weight=None, regularize=True, export=None,
                           name="Untilted_layer"):

        assert weight is not None or d_out is not None
        d_in = x.shape[1].value
        with tf.compat.v1.variable_scope(name):
            if weight is None:
                weight = tf.Variable(np.random.uniform(-np.sqrt(6/(d_in + d_out)), np.sqrt(6/(d_in + d_out)),
                                                       [d_in, d_out]), name="weight", dtype=tf.float32)  # xaiver initiation
            else:
                d_out = weight.shape[0]
                weight = tf.transpose(weight)
            bias = tf.Variable(tf.zeros(d_out), name="bias")

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
                                            regularize=True, export='layer_{}'.format(k), name="Encoder_{}".format(k+1))
            encoders.append(layer)
        return encoders

    def build_pretrain_decoders(self, encoders):
        decoders = []
        for k, encoder in enumerate(encoders[1:]):  #
            if k != 0:
                layer = self.construct_ae_layer(x=encoder['tensor'], weight=encoder['params']['w'], activationF=self.aF,
                                                regularize=False, export=None, name="Decoder_{}".format(k + 1))
            else:
                layer = self.construct_ae_layer(x=encoder['tensor'], weight=encoder['params']['w'],
                                                activationF=tf.nn.tanh,
                                                regularize=False, export=None, name="Decoder_{}".format(k + 1))
            decoders.append(layer)
        return decoders

    def build_full_decoder(self, encoders):
        x = encoders[-1]
        with tf.compat.v1.variable_scope("CDecoders"):
            for k, encoder in enumerate(encoders[:0:-1]):  #
                if k != len(encoders[:0:-1]) - 1:
                    layer = self.construct_ae_layer(x=x['tensor'], weight=encoder['params']['w'], activationF=self.aF,
                                                    regularize=False, export=None, name="CDecoder_{}".format(k+1))
                else:
                    layer = self.construct_ae_layer(x=x['tensor'], weight=encoder['params']['w'],
                                                    activationF=tf.nn.tanh,
                                                    regularize=False, export=None, name="CDecoder_{}".format(k+1))
                x = layer
        return x


class DeepDCancer:
    """
    The main supervised DeepDCancer and DeepDcCancer diagnostic model framework
    """
    def __init__(self, tp2vec, config):
        """
        :param data: (dict) training data dictionary
        :param n_iter: (int) maximum number of iterations allowed
        :param n_iter_patience: (int) tolerence of training loss no-decrease
        :return: z, xhat: (numpy.array, numpy.array) feature matrix and imputation matrix
        """






