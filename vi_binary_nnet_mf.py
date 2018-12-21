import os
import shutil
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np

from utils import Dataset


class VIBinaryNNetMF:
    def __init__(self):
        """
        Base class for binary link prediction trained by variational inference.
        """

    def build(self):

        N = self.N
        n_factors = self.n_factors
        d_pairwise = self.d_pairwise
        hidden_layer_sizes = self.hidden_layer_sizes

        self.row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        self.col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        self.val = tf.placeholder(dtype=tf.int32, shape=[None], name='val')

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

        qU_dist = ds.Normal(loc=tf.Variable(tf.random_normal([N, n_factors]), name='qU_mean'),
                            scale=tf.nn.softplus(tf.Variable(tf.ones([N, n_factors]) * init_scale), name='qU_scale_unc'),
                            name='qU_dist')

        pUp_dist = ds.Normal(loc=tf.Variable(0.0, name='pUp_mean'),
                             scale=tf.nn.softplus(tf.Variable(init_scale), name='pUp_scale_unc'),
                             name='pUp_dist')

        qUp_dist = ds.Normal(loc=tf.Variable(tf.random_normal([N, d_pairwise]), name='qUp_mean'),
                             scale=tf.nn.softplus(tf.Variable(tf.ones([N, d_pairwise]) * init_scale), name='qUp_scale_unc'),
                             name='qUp_dist')

        # compute the KL terms
        kl_terms += tf.reduce_sum(qU_dist.kl_divergence(pU_dist))  # scalar
        kl_terms += tf.reduce_sum(qUp_dist.kl_divergence(pUp_dist))

        # produce samples
        qU_samps = qU_dist.sample(self.n_samples)  # (n_samples, N, n_factors)
        qUp_samps = qUp_dist.sample(self.n_samples)  # (n_samples, N, d_pairwise)

        inputs_ = tf.concat([tf.gather(qU_samps, indices=self.row, axis=1),
                             tf.gather(qU_samps, indices=self.col, axis=1),
                             tf.gather(qUp_samps, indices=self.row, axis=1) * tf.gather(qUp_samps, indices=self.col, axis=1)
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
            # inputs_ = tf.einsum('jkl,ijl->ijk', qW_samps, inputs_) + qB_samps  # (batch_size, n_samples, layer_size)
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

        logits = tf.matmul(inputs_, qW_samps) + qB_samps[:, None, None]  # (n_samples, batch_size, layer_size) x (n_samples, layer_size, 1) --> (n_samples, batch_size, 1)
        # self.logits = tf.einsum('jk,ijk->ij', qW_samps, inputs_) + qB_samps  # (batch_size, n_samples)

        vals = tf.tile(self.val[None, :], [tf.shape(logits)[0], 1])  # (n_samples, batch_size)
        logits = tf.squeeze(logits)  # (n_samples, batch_size, 1) --> (n_samples, batch_size)
        data_loglikel = - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(vals, tf.float32),
                                                                  logits=logits)  # (n_samples, batch_size)

        self.batch_scale = tf.placeholder(dtype=tf.float32, shape=[], name='batch_scale')
        data_loglikel = self.batch_scale * tf.reduce_sum(data_loglikel, axis=1)  # (n_samples,)
        self.data_loglikel = tf.reduce_mean(data_loglikel)
        self.elbo = self.data_loglikel - kl_terms


    def train(self, N, rows, cols, miss_rows=None, miss_cols=None,
              n_factors=20, d_pairwise=1, hidden_layer_sizes=[],
              n_iterations=1000, batch_size=None, holdout_ratio=None,
              learning_rate=0.001, n_samples=10,
              root_savedir='saved', root_logdir=None,
              no_train_metric=False, seed=None):

        """
        Training routine.

        :param N: Number of nodes
        :param rows: Rows for "on" entries
        :param cols: Corresponding columns for "on" entries
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

        self.N = N
        self.n_factors = n_factors
        self.d_pairwise = d_pairwise
        self.hidden_layer_sizes = hidden_layer_sizes

        if not os.path.exists(root_savedir):
            os.makedirs(root_savedir)

        root_logdir = os.path.join(root_savedir, 'tf_logs') if root_logdir is None else root_logdir

        ###  Data handling  ###

        dataset = Dataset(N, rows, cols, miss_rows=miss_rows, miss_cols=miss_cols,
                          batch_size=batch_size, holdout_ratio=holdout_ratio, seed=seed)


        ###  Construct the TF graph  ###

        self.build()

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

        # create tensorboard summary objects
        all_vars = tf.trainable_variables()
        scalar_summaries = [tf.summary.scalar(var_.name, var_) for var_ in all_vars if len(var_.shape) == 0]
        array_summaries = [tf.summary.histogram(var_.name, var_) for var_ in all_vars if len(var_.shape) > 0]

        writer = tf.summary.FileWriter(root_logdir)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            if not no_train_metric:
                train_dict = {self.row: dataset.train[:, 0],
                              self.col: dataset.train[:, 1],
                              self.val: dataset.train[:, 2],
                              self.n_samples: 100,
                              self.batch_scale: 1.0}

            if holdout_ratio is not None:
                test_dict = {self.row: dataset.test[:, 0],
                             self.col: dataset.test[:, 1],
                             self.val: dataset.test[:, 2],
                             self.n_samples: 100,
                             self.batch_scale: 1.0}


            for iteration in range(n_iterations):

                batch = dataset.next_batch()
                sess.run(train_op, feed_dict={self.row: batch[:, 0],
                                              self.col: batch[:, 1],
                                              self.val: batch[:, 2],
                                              self.n_samples: n_samples,
                                              self.batch_scale: len(dataset.train) / len(batch)
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

                    scalar_summaries_str = sess.run(scalar_summaries)
                    array_summaries_str = sess.run(array_summaries)
                    for summary_ in scalar_summaries_str + array_summaries_str:
                        writer.add_summary(summary_, iteration)

            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()


if __name__=='__main__':

    N = 200
    X = np.random.rand(N, N) < 0.4

    from scipy.sparse import find
    rows, cols, _ = find(X)

    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved/vi_binary"
    root_logdir = os.path.join(root_savedir, "tf_logs")

    if os.path.exists(root_savedir):
        shutil.rmtree(root_savedir)

    model = VIBinaryNNetMF()
    model.train(N, rows, cols, miss_rows=None, miss_cols=None,
                n_factors=4, hidden_layer_sizes=[10, 8], d_pairwise=20,
                n_iterations=1000, batch_size=1000, holdout_ratio=0.1, learning_rate=0.01, n_samples=10,
                root_savedir=root_savedir, root_logdir=root_logdir, no_train_metric=False)

    os.system('/Users/Koa/anaconda3/bin/tensorboard --logdir=' + root_logdir)