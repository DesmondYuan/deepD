"""
This module defines main model modules
"""
import tensorflow as tf
import numpy as np
from . import utils
from .train import get_optimizer, get_mse, get_xent, get_loss


class DeepT2vec:
    """
    The main unsupervised DeepDT2vec model framework
    """
    def __init__(self, config):
        """
        :param config: (dict) training configuration
        :param seed: (int) training random seed for both data partition (numpy), params initialization (numpy)
        """
        self.config = config['unsupervised_layers']
        self.x = tf.compat.v1.placeholder("float", [None, self.config[0][0]], name='X')
        self.y = tf.compat.v1.placeholder("float", [None, config['supervised_layers'][-1][1]], name='Y')
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
            self.mse, self.loss = get_loss(self.x, self.full_decoder['tensor'], l1=self.l1, l2=self.l2)

        with tf.compat.v1.variable_scope("AEs_Loss", reuse=tf.compat.v1.AUTO_REUSE):
            self.pretrain_mses = [get_mse(x['tensor'], xhat['tensor'])
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
                # xaiver initiation
                weight = tf.Variable(np.random.uniform(-np.sqrt(6/(d_in + d_out)), np.sqrt(6/(d_in + d_out)),
                                                       [d_in, d_out]), name="weight", dtype=tf.float32)
            else:
                d_out = weight.shape[0]
                weight = tf.transpose(weight)
            bias = tf.Variable(tf.zeros(d_out), name="bias")

            params = {'w': weight, 'b': bias}
            if regularize:
                self.reg_params.append(params['w'])
                self.reg_params.append(params['b'])
            if export is not None:
                self.export_params.update({'AE_w_{}'.format(export): params['w']})
                self.export_params.update({'AE_b_{}'.format(export): params['b']})
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
        :param tp2vec: (deepD.DeepT2vec) a pretrained deep autoencoder (unsupervised)
        :param config: (dict) training configuration
        :param seed: (int) training random seed for both data partition (numpy), params initialization (numpy)
        """
        self.config = config['supervised_layers']
        self.x = tp2vec.x
        self.y = tp2vec.y
        self.x_encoded = tp2vec.encoders[-1]['tensor']
        self.true_labels = tf.compat.v1.placeholder("float", [None, self.config[-1][1]], name='Y')
        self.regularize_params, self.export_params = [], {}
        self.aF = eval(config['activation'])
        self.lr = config['learning_rate']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.batch_size = config['batch_size']

        self.yhat = self.build_layers()
        with tf.compat.v1.variable_scope("classifier_Loss", reuse=tf.compat.v1.AUTO_REUSE):
            self.xent_loss, self.loss = get_loss(self.yhat, self.y, fn=get_xent, l1=self.l1, l2=self.l2)

        self.optimizer_op_disconnected = get_optimizer(
            self.loss, self.lr, scope="DeepDCancer_opt", optimizer=config['optimizer'],
            var_list=tf.compat.v1.get_collection('variables', scope='Classifer')
        )

        self.optimizer_op_connected = get_optimizer(self.loss, self.lr, scope="DeepDcCancer_opt",
                                                    optimizer=config['optimizer'])

        self.screenshot = utils.Screenshot(self, config['n_iter_buffer'], verbose=config['verbose'],
                                           listen_freq=config['listen_freq'])

    def attach_sess(self, sess, saver):
        self.sess = sess
        self.saver = saver

    def construct_dense_layer(self, x, d_out, activationF=tf.tanh, export=None):
        d_in = x.shape[1].value
        with tf.compat.v1.variable_scope("classifier_layer"):
            # xaiver initiation
            weight = tf.Variable(np.random.uniform(-np.sqrt(6 / (d_in + d_out)), np.sqrt(6 / (d_in + d_out)),
                                                   [d_in, d_out]), name="weight", dtype=tf.float32)
            bias = tf.Variable(tf.zeros(d_out), name="bias")
            params = {'w': weight, 'b': bias}
            self.regularize_params.append(params['w'])
            self.regularize_params.append(params['b'])
            if export is not None:
                self.export_params.update({'classifier_w_{}'.format(export): params['w']})
                self.export_params.update({'classifier_b_{}'.format(export): params['b']})
            y = activationF(tf.matmul(x, weight) + bias)
        return {'tensor': y, 'params': params}

    def build_layers(self):
        x = self.x_encoded
        with tf.compat.v1.variable_scope("Classifer"):
            for k, layer_config in enumerate(self.config):
                assert x.shape[1] == layer_config[0], "Mismatched layers in supervised config."
                layer = self.construct_dense_layer(x=x, d_out=layer_config[1], activationF=tf.nn.tanh, export=k+1)
                x = layer['tensor']
        return x








