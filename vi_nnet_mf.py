import os
import shutil
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np

from utils import BatchGenerator, get_pairs


class VINNetMF:
    def __init__(self):
        """
        Base class for binary link prediction trained by variational inference.
        """

    def construct_graph(self):

        n_rows = self.n_rows
        n_cols = self.n_cols
        n_factors = self.n_factors
        d_pairwise = self.d_pairwise
        hidden_layer_sizes = self.hidden_layer_sizes

        self.row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        self.col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        self.val = tf.placeholder(dtype=tf.float32, shape=[None], name='val')

        # if self.n_side>0:
        #     self.side = tf.placeholder(dtype=tf.float32, shape=[None, self.n_side])

        self.n_samples = tf.placeholder(dtype=tf.int32, shape=[], name='n_samples')

        # initial value of any unconstrained variational scale parameters
        init_scale = 0.0001

        kl_terms = tf.constant(0.0)

        # Should the inputs be fixed parameters or random variables?
        pU_dist = ds.Normal(loc=tf.Variable(0.0, name='pU_mean'),
                            scale=tf.nn.softplus(tf.Variable(init_scale), name='pU_scale_unc'),
                            name='pU_dist')

        qU_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_rows, n_factors]), name='qU_mean'),
                            scale=tf.nn.softplus(tf.Variable(tf.ones([n_rows, n_factors]) * init_scale), name='qU_scale_unc'),
                            name='qU_dist')

        pUp_dist = ds.Normal(loc=tf.Variable(0.0, name='pUp_mean'),
                             scale=tf.nn.softplus(tf.Variable(init_scale), name='pUp_scale_unc'),
                             name='pUp_dist')

        qUp_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_rows, d_pairwise]), name='qUp_mean'),
                             scale=tf.nn.softplus(tf.Variable(tf.ones([n_rows, d_pairwise]) * init_scale), name='qUp_scale_unc'),
                             name='qUp_dist')

        pV_dist = ds.Normal(loc=tf.Variable(0.0, name='pV_mean'),
                            scale=tf.nn.softplus(tf.Variable(init_scale), name='pV_scale_unc'),
                            name='pV_dist')

        qV_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_cols, n_factors]), name='qV_mean'),
                            scale=tf.nn.softplus(tf.Variable(tf.ones([n_cols, n_factors]) * init_scale), name='qV_scale_unc'),
                            name='qV_dist')

        pVp_dist = ds.Normal(loc=tf.Variable(0.0, name='pVp_mean'),
                             scale=tf.nn.softplus(tf.Variable(init_scale), name='pVp_scale_unc'),
                             name='pVp_dist')

        qVp_dist = ds.Normal(loc=tf.Variable(tf.random_normal([n_cols, d_pairwise]), name='qVp_mean'),
                             scale=tf.nn.softplus(tf.Variable(tf.ones([n_cols, d_pairwise]) * init_scale), name='qVp_scale_unc'),
                             name='qVp_dist')

        # compute the KL terms
        kl_terms += tf.reduce_sum(qU_dist.kl_divergence(pU_dist))  # scalar
        kl_terms += tf.reduce_sum(qUp_dist.kl_divergence(pUp_dist))
        kl_terms += tf.reduce_sum(qV_dist.kl_divergence(pV_dist))  # scalar
        kl_terms += tf.reduce_sum(qVp_dist.kl_divergence(pVp_dist))

        # produce samples
        qU_samps = qU_dist.sample(self.n_samples)  # (n_samples, n_rows, n_factors)
        qUp_samps = qUp_dist.sample(self.n_samples)  # (n_samples, n_rows, d_pairwise)
        qV_samps = qV_dist.sample(self.n_samples)  # (n_samples, n_cols, n_factors)
        qVp_samps = qVp_dist.sample(self.n_samples)  # (n_samples, n_cols, d_pairwise)

        inputs_ = tf.concat([tf.gather(qU_samps, indices=self.row, axis=1),
                             tf.gather(qV_samps, indices=self.col, axis=1),
                             tf.gather(qUp_samps, indices=self.row, axis=1) * tf.gather(qVp_samps, indices=self.col, axis=1)
                             ], axis=2)  # (n_samples, batch_size, n_inputs)

        # if self.n_side>0:
        #     inputs_ = tf.concat(1, [inputs_,
        #                          tf.gather(self.side, self.row),
        #                          tf.gather(self.side, self.col)])

        # Shared p-distribution over the nnet weights and biases
        pW_dist = ds.Normal(loc=tf.constant(0.0),
                            scale=tf.nn.softplus(tf.Variable(init_scale)),
                            name='pW_dist')

        pB_dist = ds.Normal(loc=tf.constant(0.0),
                            scale=tf.nn.softplus(tf.Variable(init_scale)),
                            name='pB_dist')

        activation_fn = tf.nn.relu

        n_inputs = 2 * n_factors + d_pairwise

        self.q_layers = []  # for storage
        for layer_size in hidden_layer_sizes:

            # sample weights
            qW_mean = tf.Variable(tf.zeros([n_inputs, layer_size]))
            qW_scale = tf.nn.softplus(tf.Variable(tf.ones([n_inputs, layer_size]) * init_scale))
            qW_dist = ds.Normal(loc=qW_mean, scale=qW_scale)
            qW_samps = qW_dist.sample(self.n_samples)  # (n_samples, n_inputs, layer_size)

            # sample biases
            qB_mean = tf.Variable(tf.zeros([layer_size]))
            qB_scale = tf.nn.softplus(tf.Variable(tf.ones([layer_size]) * init_scale))
            qB_dist = ds.Normal(loc=qB_mean, scale=qB_scale)
            qB_samps = qB_dist.sample(self.n_samples)  # (n_samples, layer_size)

            inputs_ = tf.matmul(inputs_, qW_samps) + qB_samps[:, None, :] # (n_samples, batch_size, n_inputs) x (n_samples, n_inputs, layer_size) --> (n_samples, batch_size, layer_size)
            inputs_ = activation_fn(inputs_)
            n_inputs = layer_size

            # store the variables
            self.q_layers.append((qW_mean, qW_scale, qB_mean, qB_scale))

            # add up the KL terms
            kl_terms += tf.reduce_sum(qW_dist.kl_divergence(pW_dist))
            kl_terms += tf.reduce_sum(qB_dist.kl_divergence(pB_dist))


        # output layer, univariate output
        qW_mean = tf.Variable(tf.zeros([n_inputs, 1]))
        qW_scale = tf.nn.softplus(tf.Variable(tf.ones([n_inputs, 1]) * init_scale))
        qW_dist = ds.Normal(loc=qW_mean, scale=qW_scale)
        qW_samps = qW_dist.sample(self.n_samples)  # (n_samples, n_inputs, 1)

        qB_mean = tf.Variable(0.0)
        qB_scale = tf.nn.softplus(tf.Variable(init_scale))
        qB_dist = ds.Normal(loc=qB_mean, scale=qB_scale)
        qB_samps = qB_dist.sample(self.n_samples)  # (n_samples,)

        self.q_layers.append((qW_mean, qW_scale, qB_mean, qB_scale))

        kl_terms += tf.reduce_sum(qW_dist.kl_divergence(pW_dist))
        kl_terms += tf.reduce_sum(qB_dist.kl_divergence(pB_dist))

        outputs = tf.matmul(inputs_, qW_samps) + qB_samps[:, None, None]  # (n_samples, batch_size, layer_size) x (n_samples, layer_size, 1) --> (n_samples, batch_size, 1)
        outputs = tf.squeeze(outputs)  # (n_samples, batch_size)

        # Gaussian output model
        noise_std = tf.nn.softplus(tf.Variable(tf.random_normal([]), name='noise_std_unc'))
        data_loglikel = - 0.5 * tf.log(2.0 * np.pi) - tf.log(noise_std) \
                        - 0.5 * (noise_std**-2.0) * ((outputs - self.val)**2.0)  # (n_samples, batch_size)

        self.batch_scale = tf.placeholder(dtype=tf.float32, shape=[], name='batch_scale')
        data_loglikel = self.batch_scale * tf.reduce_sum(data_loglikel, axis=1)  # (n_samples,)
        self.data_loglikel = tf.reduce_mean(data_loglikel)
        self.elbo = self.data_loglikel - kl_terms


    def train(self, n_rows, n_cols, rows, cols, vals, n_factors, d_pairwise, hidden_layer_sizes,
              n_iterations, batch_size, holdout_ratio, learning_rate, n_samples,
              root_savedir, root_logdir,
              no_train_metric=False, seed=None):

        """
        Training routine.

        :param n_rows: Number of rows
        :param n_cols:
        :param rows: Rows for "on" entries
        :param cols: Corresponding columns for "on" entries
        :param vals:
        :param n_factors: Number of non-bilinear terms
        :param d_pairwise: Number of bilinear terms
        :param hidden_layer_sizes:
        :param n_iterations:
        :param batch_size:
        :param holdout_ratio:
        :param learning_rate:
        :param n_samples:
        :param root_savedir:
        :param root_logdir:
        :param no_train_metric:
        :param seed:
        :return:
        """

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_factors = n_factors
        self.d_pairwise = d_pairwise
        self.hidden_layer_sizes = hidden_layer_sizes

        if not os.path.exists(root_savedir):
            os.makedirs(root_savedir)

        ###  Data handling  ###

        pairs = np.vstack([rows, cols, vals]).T  # (3, n_obs)
        batch_generator = BatchGenerator(pairs, batch_size, holdout_ratio=holdout_ratio, seed=seed)


        ###  Construct the TF graph  ###

        self.construct_graph()

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-self.elbo)

        ###  Training  ###

        if not no_train_metric:
            train_elbo = tf.placeholder(dtype=tf.float32, shape=[], name='train_elbo')
            train_elbo_summary = tf.summary.scalar('train_elbo', train_elbo)

            train_ll = tf.placeholder(dtype=tf.float32, shape=[], name='train_ll')
            train_ll_summary = tf.summary.scalar('train_ll', train_ll)

        if holdout_ratio is not None:
            test_ll = tf.placeholder(dtype=tf.float32, shape=[], name='test_ll')
            test_ll_summary = tf.summary.scalar('test_ll', test_ll)

        writer = tf.summary.FileWriter(root_logdir)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            if not no_train_metric:
                train_dict = {self.row: batch_generator.train[:, 0],
                              self.col: batch_generator.train[:, 1],
                              self.val: batch_generator.train[:, 2],
                              self.n_samples: 100,
                              self.batch_scale: 1.0}

            if holdout_ratio is not None:
                test_dict = {self.row: batch_generator.test[:, 0],
                             self.col: batch_generator.test[:, 1],
                             self.val: batch_generator.test[:, 2],
                             self.n_samples: 100,
                             self.batch_scale: 1.0}


            for iteration in range(n_iterations):

                batch = batch_generator.next_batch()
                sess.run(train_op, feed_dict={self.row: batch[:, 0],
                                              self.col: batch[:, 1],
                                              self.val: batch[:, 2],
                                              self.n_samples: n_samples,
                                              self.batch_scale: len(batch_generator.train) / len(batch)
                                              })

                if iteration % 20 == 0:

                    print(iteration, end="")

                    if not no_train_metric:
                        train_ll_, train_elbo_ = sess.run([self.data_loglikel, self.elbo], feed_dict=train_dict)
                        train_ll_summary_str, train_elbo_summary_str = sess.run([train_ll_summary, train_elbo_summary],
                                                                                feed_dict={train_ll: train_ll_,
                                                                                           train_elbo: train_elbo_})
                        writer.add_summary(train_ll_summary_str, iteration)
                        writer.add_summary(train_elbo_summary_str, iteration)
                        print("\tTrain ELBO: %.4f" % train_elbo_, end="")
                        print("\tTrain LL: %.4f" % train_ll_, end="")

                    if holdout_ratio is not None:
                        test_ll_ = sess.run(self.data_loglikel, feed_dict=test_dict)
                        test_ll_summary_str = sess.run(test_ll_summary, feed_dict={test_ll: test_ll_})
                        writer.add_summary(test_ll_summary_str, iteration)
                        print("\tTest LL: %.4f" % test_ll_)


            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()


if __name__=='__main__':

    # create some data from the model
    n_rows = 10
    n_cols = 8
    n_factors = 6
    d_pairwise = 12

    U = np.random.randn(n_rows, n_factors) * 0.8
    V = np.random.randn(n_cols, n_factors)
    Up = np.random.randn(n_rows, d_pairwise) * 1.2
    Vp = np.random.randn(n_cols, d_pairwise) * 1.8

    n_obs = 500
    rows = np.random.choice(range(n_rows), size=n_obs, replace=True)
    cols = np.random.choice(range(n_cols), size=n_obs, replace=True)
    inputs = np.concatenate([U[rows, :], V[cols, :], Up[rows, :], Vp[cols, :]], axis=1)  # (n_obs, n_inputs)
    n_inputs = inputs.shape[1]

    relu = lambda x: np.maximum(x, 0.0)

    hidden_layer_sizes = [10, 15]
    for layer_size in hidden_layer_sizes:
        W = np.random.randn(n_inputs, layer_size) * 1.1
        B = np.random.randn(layer_size) * 1.2
        inputs = relu(np.dot(inputs, W) + B)  # (n_obs, layer_size)
        n_inputs = layer_size

    W = np.random.randn(n_inputs) * 1.1
    B = np.random.randn()
    vals = np.sum(inputs * W, axis=1) + B  # (n_obs,)

    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved/vi"
    root_logdir = os.path.join(root_savedir, "tf_logs")

    if os.path.exists(root_savedir):
        shutil.rmtree(root_savedir)

    model = VINNetMF()
    model.train(n_rows, n_cols, rows, cols, vals, n_factors=4, hidden_layer_sizes=[10, 8], d_pairwise=20,
                      n_iterations=1000, batch_size=1000, holdout_ratio=0.1, learning_rate=0.01, n_samples=10,
                      root_savedir=root_savedir, root_logdir=root_logdir, no_train_metric=False)


    os.system('/Users/Koa/anaconda3/bin/tensorboard --logdir=' + root_logdir)