import os
import shutil
# from datetime import datetime
import tensorflow as tf
import numpy as np

from utils import BatchGenerator, get_pairs


class BinaryNNetMF:
    def __init__(self):
        """
        Base class for binary link prediction. This variant considers a square link matrix, where rows and columns
        correspond to the same set of nodes and thus share the same factors/embeddings. Care should be taken considering
        that the nnet function is not symmetric.
        """

    def construct_graph(self):

        N = self.N
        n_factors = self.n_factors
        d_pairwise = self.d_pairwise
        hidden_layer_sizes = self.hidden_layer_sizes
        reg_param = self.reg_param
        l2_param = self.l2_param


        self.row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        self.col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        self.val = tf.placeholder(dtype=tf.int32, shape=[None], name='val')

        # if self.n_side>0:
        #     self.side = tf.placeholder(dtype=tf.float32, shape=[None, self.n_side])


        self.U = tf.Variable(tf.random_normal([N, n_factors]), name='U')  # node specific features
        self.Up = tf.Variable(tf.random_normal([N, d_pairwise]), name='Up')


        inputs_ = tf.concat([tf.gather(self.U, self.row),
                             tf.gather(self.U, self.col),
                             tf.gather(self.Up, self.row) * tf.gather(self.Up, self.col)
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
        self.logits = tf.contrib.layers.fully_connected(inputs_, 1, activation_fn=None)

        # probs = tf.nn.sigmoid(logits)

        self.entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.val, tf.float32),
                                                                             logits=tf.squeeze(self.logits)))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if l2_param is not None else []
        print("\nReg losses:", reg_losses)

        loss = self.entropy \
                    + reg_param * (tf.reduce_sum(tf.square(self.U))
                                    + tf.reduce_sum(tf.square(self.Up))
                                    )

        self.loss = tf.add_n([loss] + reg_losses, name='loss')



    def train(self, N, rows, cols, n_factors, d_pairwise, hidden_layer_sizes,
              n_iterations, batch_size, holdout_ratio, learning_rate, reg_param, l2_param,
              root_savedir, root_logdir,
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
        :param reg_param: Frobenius norm regularization terms for the features
        :param l2_param: L2 regularization parameter for the nnet weights
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
        self.reg_param = reg_param
        self.l2_param = l2_param

        if not os.path.exists(root_savedir):
            os.makedirs(root_savedir)


        ###  Data handling  ###

        pairs = get_pairs(N, rows, cols)
        pairs = pairs.astype(int)

        batch_generator = BatchGenerator(pairs, batch_size, holdout_ratio=holdout_ratio, seed=seed)


        ###  Construct the TF graph  ###

        self.construct_graph()

        all_vars = tf.trainable_variables()
        latent_vars = [self.U, self.Up]  # the inputs to the nnets
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
            test_xent = tf.placeholder(dtype=tf.float32, shape=[], name='test_xent')
            test_xent_summary = tf.summary.scalar('test_xent', test_xent)

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
                        test_xent_ = sess.run(self.entropy, feed_dict=test_dict)
                        test_xent_summary_str = sess.run(test_xent_summary, feed_dict={test_xent: test_xent_})
                        writer.add_summary(test_xent_summary_str, iteration)
                        print("\ttest xent: %.4f" % test_xent_)


            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()


if __name__=='__main__':

    N = 200
    X = np.random.rand(N, N) < 0.4

    from scipy.sparse import find
    rows, cols, _ = find(X)

    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved/binary"
    root_logdir = os.path.join(root_savedir, "tf_logs")

    if os.path.exists(root_savedir):
        shutil.rmtree(root_savedir)

    model = BinaryNNetMF()
    model.train(N, rows, cols, n_factors=20, hidden_layer_sizes=[20, 10], d_pairwise=20,
                      n_iterations=1000, batch_size=None, holdout_ratio=0.1, learning_rate=0.01, reg_param=0.1,
                      l2_param=None, root_savedir=root_savedir, root_logdir=root_logdir, no_train_metric=False)

    os.system('/Users/Koa/anaconda3/bin/tensorboard --logdir=' + root_logdir)