# Copyright 2018 The CSGAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the utility functions for CSGAN."""

import numpy as np
import os
import scipy.misc
import scipy.io
import cPickle
import time
from pip.utils import ensure_dir

from ops import *

FLAGS = tf.app.flags.FLAGS


def get_cs_learning_rate(global_step, cs_learning_rate=None):
    """Returns a learning rate tensor.

    Args:
        global_step: The global step variable. 
        cs_learning_rate: (optional) Initial learning rate. If not passed, the learning rate from the global flags will
        be used.

    Returns:
        A tenosr containing the learning rate.
    """

    if cs_learning_rate is None:
        cs_learning_rate = FLAGS.cs_learning_rate

    if hasattr(FLAGS, 'cs_decay_lr_iter'):  # If decay rate is set.
        if FLAGS.cs_decay_lr_iter > 0:  # If it is positive.
            return tf.train.exponential_decay(cs_learning_rate, global_step,
                                              FLAGS.cs_decay_lr_iter, 0.1,
                                              staircase=True)
        else:
            return tf.constant(FLAGS.cs_learning_rate)
    else:
        return tf.constant(FLAGS.cs_learning_rate)


def imresize(im, factor):
    """Resizes im by a factor.

    Args:
        im: Input np.ndarray image.
        factor: The resizing factor.

    Returns:
        The resized np.ndarray image.
    """

    w_h = (factor * 1.0 * np.array([im.shape[1], im.shape[0]])).astype(
        int).tolist()
    return scipy.misc.imresize(im, w_h)


def get_A_path(tf_flags=None):
    """Gets the generic path for the sampling matrix.

    Args:
        tf_flags: (optional) Configuration flags, if not set the global flags will be used.

    Returns:
        A string containing the path
    """
    if tf_flags is None:
        tf_flags = FLAGS

    ensure_dir(tf_flags.sampling_mat_dir)
    return os.path.join(tf_flags.sampling_mat_dir,
                        '{}_w{}_h{}_m{}_a{}.pckl'.format(tf_flags.dataset,
                                                         tf_flags.output_width,
                                                         tf_flags.output_height,
                                                         tf_flags.cs_num_measurements,
                                                         tf_flags.a_ind))


def get_A(flags=None):
    """Gets the saved sampling matrix.
    The sampling matrices are saved so the same one is loaded across the different experiments.

    Args:
        flags: (optional) Configuration flags, if not set the global flags will be used.

    Returns:
        The sampling matrix of type np.ndarray.
    """
    A_path = get_A_path(flags)
    if os.path.exists(A_path):
        with open(A_path, 'r') as f:
            A_val = cPickle.load(f)
    else:
        return A_path
    return A_val


def cs_estimator(gen_model, A_val=None, scope='', cs_max_update_iter=None,
                 cs_num_random_restarts=None,
                 cs_learning_rate=None, cs_bsize=None):
    """Creates a CS estimator.

    Args:
        gen_model: A trained CSGAN model for a specific sampling matrix.
        A_val: (optional) The sampling matrix.
        scope: (optional) Name of a TensorFlow scope to define the new ops and variables in.
        cs_max_update_iter: (optional) Integer number of GD updates. 
        cs_num_random_restarts: (optional) Number of random restarts.
        cs_learning_rate: (optional) The learning rate for GD.
        cs_bsize: (optional) Batch size.

    Returns:
        A function that estimates the reconstructions of its inputs.
    """

    if cs_max_update_iter is None:
        cs_max_update_iter = FLAGS.cs_max_update_iter
    if cs_num_random_restarts is None:
        cs_num_random_restarts = FLAGS.cs_num_random_restarts

    sess = gen_model.sess

    if FLAGS.is_crop:
        image_dims = [FLAGS.output_height, FLAGS.output_width, FLAGS.c_dim]
    else:
        image_dims = [FLAGS.input_height, FLAGS.input_width, FLAGS.c_dim]

    if cs_learning_rate is None:
        cs_learning_rate = FLAGS.cs_learning_rate

    if A_val is None:
        if hasattr(FLAGS, 'superres_factor'):
            A_val = get_A_superres()
        elif hasattr(FLAGS, 'inpaint_ratio') or FLAGS.dc_inpaint_ratio > 0:
            setattr(FLAGS, 'inpaint_ratio', FLAGS.dc_inpaint_ratio)
            A_val = get_A_inpaint()
        else:
            A_val = get_A()

    if FLAGS.prior_test:
        # For super resolution, first upsamples the smaller image and solve:
        # min|| up(y) - G(z) ||.
        # This gives prior for z to initialize with, and solve:
        # min|| y - AG(z)||
        cs_learning_rate = FLAGS.cs_learning_rate / FLAGS.cs_num_random_restarts
        cs_max_update_iter = FLAGS.cs_max_update_iter * FLAGS.cs_num_random_restarts
        cs_num_random_restarts = 1

    num_measurements = A_val.shape[1]
    A = tf.constant(A_val, dtype=tf.float32)

    if cs_bsize is None:
        cs_bsize = FLAGS.test_batch_size

    test_images = tf.placeholder(tf.float32,
                                 [cs_bsize, np.prod(image_dims)],
                                 name='target_xs_' + scope)
    y_batch = tf.matmul(tf.reshape(test_images, [cs_bsize, -1]), A) \
              + FLAGS.cs_noise_std * tf.random_normal(
        [cs_bsize, num_measurements])

    z_batch = tf.Variable(
        tf.random_normal([cs_bsize * cs_num_random_restarts, gen_model.z_dim]),
        name='z_batch_var_' + scope)
    zp_loss_batch = tf.reduce_mean(z_batch ** 2, 1)

    z_init_pl = tf.placeholder(tf.float32, [cs_bsize * cs_num_random_restarts,
                                            gen_model.z_dim])
    z_init_op = tf.assign(z_batch, z_init_pl)

    if cs_num_random_restarts > 1:
        y_batch = tf.tile(y_batch, [cs_num_random_restarts, 1])

    x_hat_batch = gen_model.generator(z_batch, reuse=True,
                                      batch_size=cs_bsize * cs_num_random_restarts,
                                      train=False)
    if gen_model.cs_m_and_orig:
        prob, _ = gen_model.discriminator(x_hat_batch, reuse=True,
                                          batch_size=cs_bsize * cs_num_random_restarts,
                                          train=False, orig=True)

    y_hat_batch = tf.matmul(
        tf.reshape(x_hat_batch, [int(x_hat_batch.get_shape()[0]), -1]), A,
        name='y2_batch_' + scope)

    # Define all losses.
    m_loss1_batch = tf.reduce_mean(tf.abs(y_batch - y_hat_batch), 1)
    m_loss2_batch = tf.reduce_mean((y_batch - y_hat_batch) ** 2, 1)
    zp_loss_batch = zp_loss_batch

    # Define total loss.
    m_loss1 = tf.reduce_mean(m_loss1_batch)
    m_loss2 = tf.reduce_mean(m_loss2_batch)
    zp_loss = tf.reduce_mean(zp_loss_batch)

    total_loss_batch = FLAGS.mloss1_weight * m_loss1_batch \
                       + FLAGS.mloss2_weight * m_loss2_batch \
                       + FLAGS.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(total_loss_batch)

    # Set up gradient descent.
    global_step = tf.Variable(0, trainable=False, name='global_step_' + scope)
    learning_rate = get_cs_learning_rate(global_step,
                                         cs_learning_rate=cs_learning_rate)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        opt = tf.train.AdamOptimizer(learning_rate)
        update_op = opt.minimize(total_loss, var_list=[z_batch],
                                 global_step=global_step, name='update_op')

    # Intialize and restore model parameters.
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            print var
            uninitialized_vars.append(var)
    initialize_op = tf.variables_initializer(uninitialized_vars)

    def estimator(input_test_images, z_init_val=None):
        """ This is the functional that will be used to reconstruct input 
        test images. 

        The appropriate variables of this functional is set in the parent 
        cs_estimator function. The input images will be measured with the 
        sampling matrix and then get reconstructed. 

        Args:
            input_test_images: Input images of type np.ndarray. 
            z_init_val: Initial value for zs of type np.ndarray.

        Returns:
            The reconstruction of type np.ndarray, the latent variables 
            np.ndarray, the measurements that are reconstructed of type 
            np.ndarray. 
        """

        feed_dict = {test_images: input_test_images}
        sess.run(initialize_op)
        if z_init_val is not None:
            sess.run([z_init_op], feed_dict={z_init_pl: z_init_val})

        item_inds = []  # Item inds to take care of random restarts in parallel.
        for i in range(cs_bsize):
            cur_item_inds = []
            for j in range(cs_num_random_restarts):
                inds = [i + j * cs_bsize]
                cur_item_inds.append(inds)
            cur_item_inds = np.concatenate(cur_item_inds).flatten()
            item_inds.append(cur_item_inds)

        for j in range(cs_max_update_iter):  # Gradient descent loop.
            sess.run(
                [update_op], feed_dict=feed_dict)

        y_batch_val, lr_val, total_loss_val, all_x_hat_batch_val, all_total_loss_batch_val, all_z_hat, \
        m_loss1_val, \
        m_loss2_val, \
        zp_loss_val = sess.run(
            [y_batch, learning_rate, total_loss, x_hat_batch, total_loss_batch,
             z_batch,
             m_loss1,
             m_loss2,
             zp_loss], feed_dict=feed_dict)

        logging_format = 'cs: rr{} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'

        print logging_format.format(cs_num_random_restarts, j, lr_val,
                                    total_loss_val,
                                    m_loss1_val, m_loss2_val, zp_loss_val)

        ret_zs = np.zeros([cs_bsize, gen_model.z_dim])
        ret_x_hats = np.zeros([cs_bsize] + list(all_x_hat_batch_val.shape[1:]))
        ret_ys = np.zeros([cs_bsize, num_measurements])
        best_inds = np.zeros([cs_bsize], dtype=np.int32)

        for i in range(cs_bsize):
            if item_inds[i].shape[0] == 1:
                best_inds[i] = i
            else:
                best_inds[i] = item_inds[i][
                    np.argmin(all_total_loss_batch_val[item_inds[i]])].astype(
                    int)
            ret_zs[i, :] = all_z_hat[best_inds[i]]
            ret_ys[i, :] = y_batch_val[best_inds[i]]
            ret_x_hats[i, :] = all_x_hat_batch_val[best_inds[i]]

        return [ret_x_hats, ret_zs, ret_ys]

    return estimator


def get_mses(x_hats, x, normalize=False):
    """Gets the mean squared difference between x_hats and xs.

    Args:
        x_hats: A np.ndarray of NxD, reconstructed images.
        x: A np.ndarray of NxD, original images.
        normalize: (optional) If it's set true, it will normalize the inputs to [0, 1] so that the results are
        comparable with baseline papers.

    Returns:
        An array of size N containing the squarred difference of the data points.
    """
    cur_batch_size = x.shape[0]
    if normalize:
        x = x + 1
        x = x / 2.0
        x_hats = x_hats + 1
        x_hats = x_hats / 2.0

    return np.mean((x.reshape([cur_batch_size, -1]) - x_hats.reshape(
        [cur_batch_size, -1])) ** 2, axis=1)


def get_mmse(x_hats, x, normalize=False):
    """Gets the mean MSE over the set of images, as well as the standard deviation.

    Args:
        x_hats: Reconstructed images.
        x: Original images.
        normalize: (optional) If it's set true, it will normalize the inputs to [0, 1] so that the results are
        comparable with baseline papers.

    Returns:
        Mean and Standard Deviation of MSE.
    """

    all_mse = get_mses(x_hats, x, normalize=normalize)
    return np.mean(all_mse), np.std(all_mse)


def save_x_hats(gan, ds):
    """Saves the generated images of CSGAN for a dataset.

    Args:
        gan: A CSGAN object.
        ds: A Dataset object.
    """

    x_hats = cs_test_values(gan, ds, split=FLAGS.test_results_split)
    x_hats = x_hats[0]
    x_hat_path = get_var_path(gan, 'x_hats', split=FLAGS.test_results_split,
                              cs_num_random_restarts=FLAGS.cs_num_random_restarts,
                              cs_max_update_iter=FLAGS.cs_max_update_iter,
                              cs_learning_rate=FLAGS.cs_learning_rate)
    with open(x_hat_path, 'w') as f:
        cPickle.dump(x_hats, f)
        print('X_HATS ARE SAVED to {}'.format(x_hat_path))


def save_mse(gan, ds):
    """Saves the mean squared error between the reconstructions and the original images of a dataset.

    Args:
        gan: A trained CSGAN model.
        ds: A Dataset object.

    Returns:
        MSE and standard deviation.
    """
    if FLAGS.debug:
        generic_path = 'mse_debug_iter{}_lr{}_rr{}_m{}_a{}_{}.txt'
    else:
        generic_path = 'mse_iter{}_lr{}_rr{}_m{}_a{}_{}'

        if hasattr(FLAGS, 'superres_factor'):
            generic_path = generic_path + '_super'

        if FLAGS.num_tests > 0:
            generic_path = generic_path + '_numtests_{}'.format(FLAGS.num_tests)

        generic_path = generic_path + '.txt'

    res_path = os.path.join(gan.ckpt_dir, gan.model_dir, 'results',
                            generic_path.format(FLAGS.cs_max_update_iter,
                                                FLAGS.cs_learning_rate,
                                                FLAGS.cs_num_random_restarts,
                                                FLAGS.cs_num_measurements,
                                                FLAGS.a_ind,
                                                FLAGS.test_results_split))

    if os.path.exists(
            res_path) and not FLAGS.test_recompute and not FLAGS.debug and not FLAGS.vis:
        with open(res_path) as f:
            c = f.readline().strip().split(' ')
            mse = float(c[0])
            std = float(c[1])
        print('[#] MSE: {:.4f} {}.'.format(mse, res_path))
    else:

        if not os.path.exists(os.path.dirname(res_path)):
            os.makedirs(os.path.dirname(res_path))

        x_hats = cs_test_values(gan, ds, split=FLAGS.test_results_split,
                                num_tests=FLAGS.num_tests)
        x_hats = x_hats[0]
        data_X, _, _ = ds.load(FLAGS.test_results_split)
        if FLAGS.debug:
            data_X = data_X[:FLAGS.test_batch_size]

        normalize = False

        if 'mnist' in FLAGS.dataset.lower():
            # Brings the values from [-1,1] to [0,1] so that it's comparable with the numbers in the CSGM paper.
            normalize = True

        mse, std = get_mmse(x_hats, ds.transform(data_X[:x_hats.shape[0]]),
                            normalize=normalize)

        with open(res_path, 'w') as f:
            f.write(
                '{} {} {} \n'.format(str(mse), str(std), str(int(time.time()))))

        print(
            '[*] Appended MSE: {} +- of {} to {}.'.format(mse, std, gan.model_dir,
                                                          res_path))

    return mse, std


def save_image(image, path):
    """Save an image as a png file."""
    min_val = image.min()
    if min_val < 0:
        image = image + min_val

    image = (image.squeeze() * 1.0 / image.max()) * 255
    image = image.astype(np.uint8)

    scipy.misc.imsave(path, image)
    print('[#] Image saved {}.'.format(path))


def get_var_path(gen_model, var_name='z_hats', split='train',
                 cs_num_random_restarts=10, cs_max_update_iter=100,
                 cs_learning_rate=1.0):
    """Gets the name of a variable that's going to be cached."""
    exp_dir = os.path.join(FLAGS.checkpoint_dir, gen_model.model_dir,
                           'cache_iter_{}'.format(cs_max_update_iter))

    generic_path = os.path.join(exp_dir, '{}_{}_lr{}_rr{}_m{}_a{}_c{}.pkl')
    return generic_path.format(split, var_name, cs_learning_rate,
                               cs_num_random_restarts,
                               FLAGS.cs_num_measurements, 0,
                               gen_model.counter)


def cs_test_values(gen_model, dataset=None, split='train', num_tests=-1):
    """Test a CSGAN model on the whole dataset and extract reconstructions and latent variables.

    Args:
        gen_model: a CSGAN model. 
        dataset: (optional) A Dataset object.
        split: (optional) Dataset split to reconstruct.
        num_tests: (optional) The number of test samples to perform reconstruction on.
    """

    data_X, data_Y, data_ids = dataset.load(split)
    if num_tests > -1:
        data_X = data_X[:num_tests]

    exp_dir = os.path.join(gen_model.ckpt_dir, 'cache_iter_{}'.format(FLAGS.cs_max_update_iter))
    if FLAGS.debug:
        exp_dir += '_debug'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    save_arr = ['data_z_hats', 'data_measurements']
    generic_path = os.path.join(exp_dir, '{}_{}_lr{}_rr{}_m{}_a{}_c{}')
    if num_tests > 0:
        generic_path += '_numtests_{}'.format(num_tests)

    generic_path += '.pkl'

    compute_flag = True
    var_paths = []
    for var_name in save_arr:
        if not hasattr(gen_model, 'counter'):  # If it has not been initialized.
            gen_model.load()

        var_path = generic_path.format(split, var_name.replace('data_', ''),
                                       FLAGS.cs_learning_rate,
                                       FLAGS.cs_num_random_restarts,
                                       FLAGS.cs_num_measurements, FLAGS.a_ind,
                                       gen_model.counter)
        print var_path
        var_paths.append(
            var_path)

    ret_list = []
    if os.path.exists(var_paths[0]) and not FLAGS.test_recompute:  # Cache
        try:
            for var_path in var_paths:
                with open(var_path) as f:
                    d = cPickle.load(f)
                    ret_list.append(d)
                print('Loaded {} from {}'.format(split, var_name, var_path))
            compute_flag = False

        except Exception, e:
            print('WARNING: Could not load from cache, recomputing. Err: {}.'.format(str(e)))

    if compute_flag or FLAGS.test_recompute:  # Don't care about cache.
        transform_func = lambda x: dataset.transform(x, FLAGS.input_transform_type)
        data_x_hats, data_z_hats, data_measurements = get_reconstructions(gen_model, data_X, transform=transform_func)
        x_hats = data_x_hats
        ret_list = [data_x_hats, data_z_hats, data_measurements]
        for j, item in enumerate(save_arr):
            print var_paths
            file_path = var_paths[j]
            with open(file_path, 'w') as f:
                cPickle.dump(eval(item), f, cPickle.HIGHEST_PROTOCOL)
            print('Saved {} to {}.'.format(item, file_path))
    else:
        x_hats = gen_model.get_x_hats(ret_list[0])

        ret_list = [x_hats] + ret_list

    if FLAGS.vis and FLAGS.debug:  # For visualizing the reconstructions.
        dir_path = os.path.join('debug', gen_model.model_dir)
        ensure_dir(dir_path)
        for i in range(x_hats.shape[0]):
            im_size = data_X[0:1].shape[1:]
            save_image(dataset.inv_transform(x_hats[i],
                                             type=FLAGS.input_transform_type),
                       os.path.join(dir_path, 'x_test_{}_hat_{}.png'.format(i, 'loaded')))
            save_image(data_X[i],
                       os.path.join(dir_path, 'x_test_{}_{}.png'.format(i, 'original')))
            if hasattr(FLAGS, 'inpaint_ratio'):
                save_image(dataset.inv_transform(
                    data_measurements.reshape(
                        [data_measurements.shape[0]] + list(im_size)),
                    FLAGS.input_transform_type),
                    'x_test_{}_inpaint.png'.format(i))

            if hasattr(FLAGS, 'superres_factor'):
                save_image(dataset.inv_transform(
                    data_measurements.reshape([data_measurements.shape[0],
                                               im_size[0] / FLAGS.superres_factor,
                                               im_size[1] / FLAGS.superres_factor,
                                               im_size[2]]), FLAGS.input_transform_type),
                    'x_test_{}_superres.png'.format(i))

    return ret_list


def get_reconstructions(gen_model, X, transform=None):
    """Solves argmin_z || gen_model.A * gen_model.G(z) - Y ||
    where Y = gen_model.A * X_test + noise
    z_hat is the solution of SGD after FLAGS.cs_max_number_steps

    Args:
        gen_model: A CSGAN model. 
        X: An instance of Dataset or LazyDataset class.
        transform: A functional that transforms the read Xs. 
    """

    test_batch_size = FLAGS.test_batch_size

    estimator = cs_estimator(gen_model, scope='test_est', cs_bsize=test_batch_size)

    if hasattr(FLAGS, 'superres_factor') and FLAGS.prior_test:
        # Prior initialization for super-resolution (not included in the paper).
        super_prior = cs_estimator(gen_model, A_val=np.eye(FLAGS.n_input),
                                   scope='test_est_super',
                                   cs_max_update_iter=50,
                                   cs_num_random_restarts=5,
                                   cs_learning_rate=1.0)

    if not gen_model.load(FLAGS.checkpoint_dir):
        raise RuntimeError(
            "[!] No trained model. Train a model first, then run test mode.")

    num_test_data = len(X)
    print('[#] Num test: {}.'.format(num_test_data))

    data_z_hats = []
    data_x_hats = []
    data_measurements = []

    num_test_data = len(X)
    num_batches = num_test_data // test_batch_size
    if num_test_data % test_batch_size != 0:
        num_batches = num_batches + 1

    start = 0
    if FLAGS.debug:
        num_batches = 1
    means = []

    for i in range(start, num_batches):
        # Extract reconstructions for every image.
        test_batch_X = X[i * test_batch_size:min((i + 1) * test_batch_size, len(X))]
        test_batch_X = test_batch_X.reshape([-1, np.prod(test_batch_X.shape[1:]).astype(np.int32)])
        im_size = test_batch_X.shape
        rem = test_batch_size - im_size[0]
        if rem > test_batch_size:
            dummies = np.zeros([rem] + list(im_size[1:]), dtype=test_batch_X.dtype)
            test_batch_X = np.concatenate([test_batch_X, dummies])

        if transform is not None:
            test_batch_X = transform(test_batch_X)

        if FLAGS.prior_test:
            [_, z_init, _, _] = super_prior(test_batch_X)
        else:
            z_init = None

        x_hats_batch, data_z_hats_batch, y_batch = estimator(test_batch_X, z_init_val=z_init)

        normalize = False
        if 'mnist' in FLAGS.dataset:
            # To get comparable results with the CSGM paper.
            normalize = True

        means.append(
            get_mses(test_batch_X, x_hats_batch, normalize=normalize))

        data_x_hats.append(x_hats_batch)
        data_z_hats.append(data_z_hats_batch)
        data_measurements.append(y_batch)

        print('{}/{} batches {}/{} images'.format(i + 1, num_batches, (i + 1) * test_batch_size, num_test_data))

    data_x_hats = np.concatenate(data_x_hats)[:len(X)]
    data_z_hats = np.concatenate(data_z_hats)[:len(X)]
    data_measurements = np.concatenate(data_measurements)[:len(X)]

    return data_x_hats, data_z_hats, data_measurements


def get_A_superres():
    """The super-resolution matrix.

    Returns:
        The super-resolution subsampling matrix: np.ndarray of type np.int
    """

    factor = FLAGS.superres_factor
    A = np.zeros((int(FLAGS.n_input / (factor ** 2)), FLAGS.n_input))
    l = 0
    for i in range(FLAGS.image_shape[0] / factor):
        for j in range(FLAGS.image_shape[1] / factor):
            for k in range(FLAGS.image_shape[2]):
                a = np.zeros(FLAGS.image_shape)
                a[factor * i:factor * (i + 1), factor * j:factor * (j + 1),
                k] = 1
                A[l, :] = np.reshape(a, [1, -1])
                l += 1

    # Make sure that the norm of each row of A is hparams.n_input.
    A = np.sqrt(FLAGS.n_input / (factor ** 2)) * A
    assert all(np.abs(np.sum(A ** 2, 1) - FLAGS.n_input) < 1e-6)

    return A.T


def get_l2_loss(image1, image2):
    """Get L2 loss between the two images."""

    assert image1.shape == image2.shape
    return np.mean((image1 - image2) ** 2)


def get_measurement_loss(x_hat, A, y):
    """Get measurement loss of the estimated image."""

    if A is None:
        y_hat = x_hat
    else:
        y_hat = np.matmul(x_hat, A)
    assert y_hat.shape == y.shape
    return np.mean((y - y_hat) ** 2)


def get_inpaint_mask():
    """Generates the input mask.

    Returns:
        The mask.
    """

    image_size = FLAGS.image_shape[0]
    FLAGS.inpaint_size = np.ceil(FLAGS.inpaint_ratio * image_size).astype(np.int32)
    margin = (image_size - FLAGS.inpaint_size) / 2
    mask = np.ones(FLAGS.image_shape)
    mask[margin:margin + FLAGS.inpaint_size,
    margin:margin + FLAGS.inpaint_size] = 0
    return mask


def get_A_inpaint():
    """Gets the inpainting matrix.

    Returns: 
        The np.ndarray of the A matrix.
    """

    mask = get_inpaint_mask()
    mask = mask.reshape(1, -1)
    A = np.eye(np.prod(mask.shape)) * np.tile(mask, [np.prod(mask.shape), 1])
    A = np.asarray([a for a in A if np.sum(a) != 0])

    # Make sure that the norm of each row of A is hparams.n_input
    A = np.sqrt(FLAGS.n_input) * A
    assert all(np.abs(np.sum(A ** 2, 1) - FLAGS.n_input) < 1e-6)

    return A.T


class DummyWriter(object):
    """This class is used to disable the Tensorboard's summary writer."""

    def write(self, *args, **arg_dicts):
        pass

    def add_summary(self, summary_str, counter):
        pass


def generate_A():
    """Generates the A matrix if it doesn't exist already."""
    FLAGS = tf.app.flags.FLAGS
    A_path = get_A_path()
    if os.path.exists(A_path):
        print('[#] Measurement matrix already exists.')
        return

    print '[#] Generating a new sampling matrix.'
    A_val = np.random.randn(FLAGS.n_inputs, FLAGS.cs_num_measurements)
    with open(A_path, 'w') as f:
        cPickle.dump(A_val, f, cPickle.HIGHEST_PROTOCOL)
    print '[*] Generated a new sampling matrix with {} rows and {} cols. Saved in: {}' \
        .format(FLAGS.n_inputs, FLAGS.cs_num_measurements, A_path)
