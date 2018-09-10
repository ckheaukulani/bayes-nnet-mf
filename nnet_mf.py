import os
import shutil
# from datetime import datetime
import tensorflow as tf
import numpy as np

from utils import BatchGenerator


class NNetMF:
    def __init__(self):
        """
        Base class for generic neural network matrix factorization. This should largely replicate the inference procedure
        in Karolina and Dan's paper.
        """

    def construct_graph(self):

        n_rows = self.n_rows
        n_cols = self.n_cols
        n_factors = self.n_factors
        d_pairwise = self.d_pairwise
        hidden_layer_sizes = self.hidden_layer_sizes
        reg_param = self.reg_param
        l2_param = self.l2_param


        self.row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        self.col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        self.val = tf.placeholder(dtype=tf.float32, shape=[None], name='val')

        # if self.n_side>0:
        #     self.side = tf.placeholder(dtype=tf.float32, shape=[None, self.n_side])


        self.U = tf.Variable(tf.random_normal([n_rows, n_factors]), name='U')  # node specific features
        self.Up = tf.Variable(tf.random_normal([n_rows, d_pairwise]), name='Up')
        self.V = tf.Variable(tf.random_normal([n_cols, n_factors]), name='V')  # node specific features
        self.Vp = tf.Variable(tf.random_normal([n_cols, d_pairwise]), name='Vp')

        inputs_ = tf.concat([tf.gather(self.U, self.row),
                             tf.gather(self.V, self.col),
                             tf.gather(self.Up, self.row) * tf.gather(self.Vp, self.col)
                             ], axis=1)  # (batch_size, n_inputs)

        # if self.n_side>0:
        #     inputs_ = tf.concat(1, [inputs_,
        #                          tf.gather(self.side, self.row),
        #                          tf.gather(self.side, self.col)])

        activation_fn = tf.nn.relu

        weights_regularizer = tf.contrib.layers.l2_regularizer(l2_param) if l2_param is not None else None

        for layer_size in hidden_layer_sizes:
            inputs_ = tf.contrib.layers.fully_connected(inputs_, layer_size, activation_fn=activation_fn,
                                                        weights_regularizer=weights_regularizer)

        # output layer
        self.outputs = tf.contrib.layers.fully_connected(inputs_, 1, activation_fn=None)

        self.sse = tf.reduce_sum((self.val - self.outputs)**2.0)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if l2_param is not None else []
        print("\nReg losses:", reg_losses)

        loss = self.sse \
                    + reg_param * (tf.reduce_sum(tf.square(self.U))
                                   + tf.reduce_sum(tf.square(self.Up))
                                   + tf.reduce_sum(tf.square(self.V))
                                   + tf.reduce_sum(tf.square(self.Vp))
                                   )

        self.loss = tf.add_n([loss] + reg_losses, name='loss')



    def train(self, n_rows, n_cols, rows, cols, vals, n_factors, d_pairwise, hidden_layer_sizes,
              n_iterations, batch_size, holdout_ratio, learning_rate, reg_param, l2_param,
              root_savedir, root_logdir,
              no_train_metric=False, seed=None):

        """
        Training routine.

        :param n_rows: Number of rows
        :param n_cols: Number of cols
        :param rows:
        :param cols:
        :param vals:
        :param n_factors: Number of non-bilinear terms
        :param d_pairwise: Number of bilinear terms
        :param hidden_layer_sizes:
        :param n_iterations:
        :param batch_size:
        :param holdout_ratio:
        :param learning_rate:
        :param reg_param: Frobenius norm regularization terms for the features
        :param l2_param: L2 regularization parameter for the nnet weights
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
        self.reg_param = reg_param
        self.l2_param = l2_param

        if not os.path.exists(root_savedir):
            os.makedirs(root_savedir)

        ###  Data handling  ###

        # here we only train on positive examples, so all pairs are only the "on" values
        pairs = np.vstack([rows, cols, vals]).T  # (3, n_obs)
        batch_generator = BatchGenerator(pairs, batch_size, holdout_ratio=holdout_ratio, seed=seed)


        ###  Construct the TF graph  ###

        self.construct_graph()

        all_vars = tf.trainable_variables()
        latent_vars = [self.U, self.V, self.Up, self.Vp]  # the inputs to the nnets
        nnet_vars = [x for x in all_vars if x not in latent_vars]  # the nnet variables

        print("\nlatent vars:", latent_vars)
        print("\nnnet vars:", nnet_vars)

        train_lvars = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=latent_vars)
        train_nnet = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=nnet_vars)

        ###  Training  ###

        if not no_train_metric:
            train_loss = tf.placeholder(dtype=tf.float32, shape=[], name='train_loss')
            train_loss_summary = tf.summary.scalar('train_loss', train_loss)

        if holdout_ratio is not None:
            test_mse = tf.placeholder(dtype=tf.float32, shape=[], name='test_mse')
            test_mse_summary = tf.summary.scalar('test_mse', test_mse)

        # now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # logdir = "{}/run-{}/".format(root_logdir, now)
        writer = tf.summary.FileWriter(root_logdir)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            if not no_train_metric:
                train_dict = {self.row: batch_generator.train[:, 0],
                              self.col: batch_generator.train[:, 1],
                              self.val: batch_generator.train[:, 2]}

            if holdout_ratio is not None:
                test_dict = {self.row: batch_generator.test[:, 0],
                             self.col: batch_generator.test[:, 1],
                             self.val: batch_generator.test[:, 2]}


            for iteration in range(n_iterations):

                batch = batch_generator.next_batch()
                batch_dict = {self.row: batch[:, 0],
                              self.col: batch[:, 1],
                              self.val: batch[:, 2]}

                # alternate between optimizing inputs and nnet vars
                sess.run(train_lvars, feed_dict=batch_dict)
                sess.run(train_nnet, feed_dict=batch_dict)

                if iteration % 20 == 0:

                    print(iteration, end="")

                    if not no_train_metric:
                        train_loss_ = sess.run(self.loss, feed_dict=train_dict)
                        train_loss_summary_str = sess.run(train_loss_summary, feed_dict={train_loss: train_loss_})
                        writer.add_summary(train_loss_summary_str, iteration)
                        print("\ttrain loss: %.4f" % train_loss_, end="")


                    if holdout_ratio is not None:
                        test_sse_ = sess.run(self.sse, feed_dict=test_dict)
                        test_mse_ = test_sse_ / len(batch_generator.test)
                        test_mse_summary_str = sess.run(test_mse_summary, feed_dict={test_mse: test_mse_})
                        writer.add_summary(test_mse_summary_str, iteration)
                        print("\ttest mse: %.4f" % test_mse_)


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

    # now train
    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved"
    root_logdir = os.path.join(root_savedir, "tf_logs")

    if os.path.exists(root_savedir):
        shutil.rmtree(root_savedir)

    model = NNetMF()
    model.train(n_rows, n_cols, rows, cols, vals, n_factors=20, hidden_layer_sizes=[20, 10], d_pairwise=20,
                      n_iterations=1000, batch_size=None, holdout_ratio=0.1, learning_rate=0.01, reg_param=0.1,
                      l2_param=None, root_savedir=root_savedir, root_logdir=root_logdir, no_train_metric=False)


    os.system('/Users/Koa/anaconda3/bin/tensorboard --logdir=' + root_logdir)