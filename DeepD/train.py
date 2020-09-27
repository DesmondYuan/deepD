"""
This module defines core training functions for DeepD
"""
import tensorflow as tf
import numpy as np
import time


def get_mse(x, xhat):
    mse = tf.reduce_mean(tf.square(x - xhat), name='MSE')
    return mse


def get_xent(x, xhat):
    xent = tf.losses.softmax_cross_entropy(x, xhat)
    return xent


def get_loss(x, xhat, fn=get_mse, l1=0, l2=0, var_list=None):
    if var_list is None:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    n_params = sum([np.prod(p.shape) for p in var_list]).value
    l1 = l1 * tf.reduce_sum([tf.reduce_sum(tf.abs(p)) for p in var_list]) / n_params
    l2 = l2 * tf.reduce_sum([tf.reduce_sum(tf.square(p)) for p in var_list]) / n_params
    raw_loss = fn(x, xhat)
    full_loss = tf.add(raw_loss, l1 + l2, name='loss')
    return raw_loss, full_loss


def session_setup():
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='initialization'):
        print(i)
    tf.summary.FileWriter("tensorboard", sess.graph)
    return sess, saver


def session_init(sess, seed):
    sess.run(tf.compat.v1.global_variables_initializer())
    np.random.seed(seed), tf.compat.v1.set_random_seed(seed)


def pretrain(model, data, n_iter=1000, n_iter_patience=100):
    """
    For layerwise pretraining of the autoencoders

    Args
        :param model: (deepD.DeepT2cec) Constructed autoencoder models
        :param data: (dict) training data dictionary
        :param n_iter: (int) maximum number of iterations allowed
        :param n_iter_patience: (int) tolerence of training loss no-decrease

    Mutates:
        model: (Deep.DeepT2vec)
    """

    sess, screenshot = model.sess, model.screenshot
    if model.screenshot:
        print("[Training] Pretrained params detected. Skipping...")
        screenshot.load_model("Tp2vec_layer_wise_retrain.ckpt")
        return 1

    x_train_gold, x_valid_gold, x_test_gold = (data[key]['value'] for key in ['train', 'valid', 'test'])
    # Training on train set batches with early stopping on valid set batched
    for k, (mse, opt_op) in enumerate(zip(model.pretrain_mses, model.pretrain_optimizer_ops)):
        print('[Training] Pre-training on train set at {}...'.format(opt_op[1].name))
        n_unchanged = 0
        idx_iter = 0
        while True:
            if idx_iter > n_iter or n_unchanged > n_iter_patience:
                break
            t0 = time.clock()
            pos_train = np.random.choice(range(x_train_gold.shape[0]), model.pretrain_batch_size)
            pos_valid = np.random.choice(range(x_valid_gold.shape[0]), model.pretrain_batch_size)
            _, mse_train_i = sess.run((opt_op[1], mse), feed_dict={model.x: x_train_gold[pos_train]})
            loss_train_i = mse_train_i
            # record training
            mse_valid_i = sess.run(mse, feed_dict={model.x: x_valid_gold[pos_valid]})
            loss_valid_i = mse_valid_i
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

        print('[Training] Saving pre-train results at layer {}...'.format(k + 1))
        screenshot.save_model("models/Tp2vec_layer_wise_retrain.ckpt")
        screenshot.save_params()
        sess.run(tf.variables_initializer(opt_op[0].variables()))  # refresh optimizer states
        screenshot.reset()  # refresh best_loss saved in screenshot


def train(model, optimizer_op, data, full_loss, raw_loss, output, n_iter=1000, n_iter_patience=100, model_name="model"):
    """
    Core training function for all DeepD models

    Args
        :param model: (object) Constructed DeepD models
        :param optimizer_op: (tf.train.Optimizer, tf.train.Optimizer.minimize()) Tensorflow operation for each iteration
        :param data: (dict) training data dictionary
        :param raw_loss: (tf.Tensor) graph loss to optimize, e.g. Mean Squared Error for autoencoders and
                        Cross Entropy Loss for classifiers
        :param full_loss: (tf.Tensor) total graph loss with raw_loss and regularization penalty
        :param n_iter: (int) maximum number of iterations allowed
        :param n_iter_patience: (int) tolerence of training loss no-decrease
        :param output: (list of tf.Tensor) desired outputs tensors in a list
        :param model_name: filename for model saving (default: 'model'). Overiden if using a low verbose.

    Returns
        A list of numpy.array as specified in the param "output"
    """

    sess, screenshot = model.sess, model.screenshot
    n_unchanged = 0
    idx_iter = 0
    x_train_gold, x_valid_gold, x_test_gold = (data[key]['value'] for key in ['train', 'valid', 'test'])
    y_train_gold, y_valid_gold, y_test_gold = (data[key]['class_annot'] for key in ['train', 'valid', 'test'])
    sampler_train, sampler_valid = (data[key]['p_sampler'] for key in ['train', 'valid'])

    # Training on train set batches with early stopping on valid set batched
    print('[Training] Training on train set...')
    while True:
        if idx_iter > n_iter or n_unchanged > n_iter_patience:
            break
        t0 = time.clock()

        pos_train = np.random.choice(range(x_train_gold.shape[0]), model.batch_size, p=sampler_train)
        pos_valid = np.random.choice(range(x_valid_gold.shape[0]), model.batch_size, p=sampler_valid)
        _, loss_train_i, mse_train_i = sess.run((optimizer_op[1], full_loss, raw_loss), feed_dict={
            model.x: x_train_gold[pos_train], model.y: y_train_gold[pos_train]})

        # record training
        loss_valid_i, mse_valid_i = sess.run((full_loss, raw_loss), feed_dict={model.x: x_valid_gold[pos_valid],
                                                                               model.y: y_valid_gold[pos_valid]})
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
    loss_valid_i, mse_valid_i = sess.run((full_loss, raw_loss),
                                         feed_dict={model.x: x_valid_gold, model.y: y_valid_gold})
    screenshot.log(filename="training.log", iteration=(-1, -1),
                   unchanged=(-1, -1), t=time.clock() - t0,
                   loss=(np.nan, loss_valid_i, np.nan, mse_valid_i, np.nan))

    # Evaluation on test set
    print('[Training] Evaluating on test set... {}'.format(x_test_gold.shape))
    t0 = time.clock()
    mse_test = sess.run(raw_loss, feed_dict={model.x: x_test_gold, model.y: y_test_gold})
    screenshot.log(filename="training.log", iteration=(-1, -1),
                   unchanged=(-1, -1), t=time.clock() - t0,
                   loss=(np.nan, np.nan, np.nan, np.nan, mse_test))

    # Save model
    outputs = sess.run(output, feed_dict={model.x: data['test']['value'], model.y: data['test']['class_annot']})
    screenshot.save_params()
    screenshot.save_model('models/' + model_name + '.ckpt')
    sess.run(tf.variables_initializer(optimizer_op[0].variables()))  # refresh optimizer states
    screenshot.reset()  # refresh best_loss saved in screenshot
    return outputs


def get_optimizer(loss_in, lr, optimizer="tf.compat.v1.train.AdamOptimizer({})", var_list=None, scope="Optimization"):
    if var_list is None:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        opt = eval(optimizer.format(lr))
        opt_op = opt.minimize(loss_in, var_list=var_list)
    print("[Construct] Successfully generated an operation {} for optimizing: {}.".format(opt_op.name, loss_in))
    return opt, opt_op
