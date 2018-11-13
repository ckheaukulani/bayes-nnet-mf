import os
import sys
import pickle
import time
import numpy as np
import tensorflow as tf

sys.path.extend(['/Users/Koa/github-repos/han-core'
                ])
from sentence_level import SentenceLevel
from word_level import WordLevel

import data_handler

# Turn on eager execution in Tensorflow
# tf.enable_eager_execution()

class NipsHanNetwork:
    def __init__(self, word_embedding_size, word_encoder_size, word_attention_size,
                 sentence_encoder_size, sentence_attention_size, n_factors, d_pairwise, hidden_layer_sizes):
        """

        :param word_embedding_size:
        :param word_encoder_size:
        :param word_attention_size:
        :param sentence_encoder_size:
        :param sentence_attention_size:
        :param n_factors:
        :param d_pairwise:
        :param hidden_layer_sizes:
        """
        self.word_embedding_size = word_embedding_size
        self.word_encoder_size = word_encoder_size
        self.word_attention_size = word_attention_size
        self.sentence_encoder_size = sentence_encoder_size
        self.sentence_attention_size = sentence_attention_size

        self.n_factors = n_factors
        self.d_pairwise = d_pairwise
        self.hidden_layer_sizes = hidden_layer_sizes

    def build(self, vocabulary_size, max_num_docs, n_authors, reg_param, l2_param):
        """
        Build the Tensorflow graph.

        :param vocabulary_size:
        :param max_num_docs: int, the maximum number of documents for any one author
        :param n_authors:
        :param reg_param:
        :param l2_param:
        :return:
        """

        ###  BUILD THE DOCUMENT MODEL  ###

        # first create the word level model, which also creates placeholders for data
        self.word_model = WordLevel(vocabulary_size=vocabulary_size,
                                    word_embedding_size=self.word_embedding_size,
                                    word_encoder_size=self.word_encoder_size,
                                    word_attention_size=self.word_attention_size
                                    )

        # the sentence vectors from the word model are passed into the sentence level model
        self.sentence_model = SentenceLevel(sentence_vectors=self.word_model.sentence_vectors,
                                            sentence_encoder_size=self.sentence_encoder_size,
                                            sentence_attention_size=self.sentence_attention_size
                                            )

        # Map the constructed document vectors to become features for their authors; each author's feature is a sum of
        # their corresponding document vectors.
        # This is rather complicated and ugly without control flow. Not sure if there's a better way to do it with graph
        # construction... or maybe try to switch to TF eager execution?
        self.papers_by_author_batchind = tf.placeholder(dtype=tf.int32,
                                                        shape=[None, max_num_docs],
                                                        name='papers_by_author_batchind')
            # padded array representing a list of lists; for unique author in this document-batch, give the list of
            # 0-indices into the document-batch that picks out the papers corresponding to that author

        # follow a similar strategy to dynamic RNNs to pick out lists of variable sizes
        self.paper_num_by_author = tf.placeholder(dtype=tf.int32, shape=[None], name='paper_num_by_author')
            # the number of papers by each author in 'self.papers_by_author_batchind'

        print("doc vectors:", self.sentence_model.document_vectors)

        def agg_embeddings(x):
            """

            :param author_papers_batchind:
            :return:
            """
            author_papers_batchind = x[0]
            len_ = x[1]
            author_papers_batchind = author_papers_batchind[:len_]
            author_doc_embeddings = tf.gather(self.sentence_model.document_vectors, indices=author_papers_batchind)
            author_embedding = tf.reduce_sum(author_doc_embeddings, axis=0)
            return author_embedding

        # tf.map_fn returns the same type as 'elems', unless keyword 'dtype' specifies otherwise
        author_embeddings = tf.map_fn(fn=agg_embeddings,
                                      elems=(self.papers_by_author_batchind,
                                             self.paper_num_by_author),
                                      dtype=tf.float32)  # (n_batch_authors, embedding_size)

        # now map these to the entries in the row and column
        self.author_to_row = tf.placeholder(dtype=tf.int32, shape=[None], name='author_to_row')
        self.author_to_col = tf.placeholder(dtype=tf.int32, shape=[None], name='author_to_col')

        row_embeddings = tf.gather(author_embeddings, indices=self.author_to_row)  # (edge-batch_size, embedding_size)
        col_embeddings = tf.gather(author_embeddings, indices=self.author_to_col)  # (edge-batch_size, embedding_size)

        ###  BUILD THE NETWORK MODEL  ###

        self.row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        self.col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        self.val = tf.placeholder(dtype=tf.int32, shape=[None], name='val')

        self.U = tf.Variable(tf.random_normal([n_authors, self.n_factors]), name='U')
        self.Up = tf.Variable(tf.random_normal([n_authors, self.d_pairwise]), name='Up')

        inputs_ = tf.concat([tf.gather(self.U, indices=self.row),
                             tf.gather(self.U, indices=self.col),
                             tf.gather(self.Up, indices=self.row) * tf.gather(self.Up, indices=self.col),
                             row_embeddings,
                             col_embeddings
                             ], axis=-1)  # (edge-batch_size, n_inputs)

        activation_fn = tf.nn.relu

        weights_regularizer = tf.contrib.layers.l2_regularizer(l2_param) if l2_param is not None else None

        for layer_size in self.hidden_layer_sizes:
            inputs_ = tf.contrib.layers.fully_connected(inputs_, layer_size, activation_fn=activation_fn,
                                                        weights_regularizer=weights_regularizer)

        # output layer
        self.logits = tf.contrib.layers.fully_connected(inputs_, 1, activation_fn=None)

        self.entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.val, tf.float32),
                                                                             logits=tf.squeeze(self.logits)))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if l2_param is not None else []
        print("\nReg losses:", reg_losses)

        loss = self.entropy \
               + reg_param * (tf.reduce_sum(tf.square(self.U))
                              + tf.reduce_sum(tf.square(self.Up))
                              )

        self.loss = tf.add_n([loss] + reg_losses, name='loss')


    def train(self, adj_matrix, documents, papers_by_author,
              n_iterations, batch_size, learning_rate, reg_param, l2_param,
              root_savedir, root_logdir,
              holdout_ratio=None, no_train_metric=False,
              seed=None):
        """
        Training routine.

        :param adj_matrix:
        :param documents:
        :param papers_by_authors: dict like {<author_id>: [<paper_id>, ...], ...}
        :param n_iterations:
        :param batch_size:
        :param learning_rate:
        :param reg_param:
        :param l2_param:
        :param root_savedir:
        :param root_logdir:
        :param no_train_metric:
        :param seed:
        :return:
        """

        # print("Tensorflow eagerly executing?", tf.executing_eagerly())

        dataset = data_handler.Dataset(adj_matrix, documents, papers_by_author,
                                       batch_size=batch_size, holdout_ratio=holdout_ratio, seed=seed)

        n_authors = dataset.n_authors
        vocabulary_size = dataset.vocabulary_size
        max_num_docs = dataset.max_num_docs

        # # keep track of the global step when storing the TF session
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #
        # # training flag for batch normalization and dropout
        # self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')


        ###  Construct the TF graph  ###

        self.build(vocabulary_size, max_num_docs, n_authors, reg_param, l2_param)

        all_vars = tf.trainable_variables()
        latent_vars = [self.U, self.Up]  # the inputs to the nnets
        nnet_vars = [x for x in all_vars if x not in latent_vars]  # the nnet variables for the

        print("\nlatent vars:", latent_vars)
        print("\nnnet vars:", nnet_vars)
        print("\nword embeddings:", self.word_model.word_embeddings)

        train_lvars = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=latent_vars)
        train_nnet = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=nnet_vars)
        train_embeddings = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, var_list=[self.word_model.word_embeddings])

        ###  Training  ###

        if not no_train_metric:
            train_loss = tf.placeholder(dtype=tf.float32, shape=[], name='train_loss')
            train_loss_summary = tf.summary.scalar('train_loss', train_loss)

        if holdout_ratio is not None:
            test_xent = tf.placeholder(dtype=tf.float32, shape=[], name='test_xent')
            test_xent_summary = tf.summary.scalar('test_xent', test_xent)

        writer = tf.summary.FileWriter(root_logdir)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            t_start = time.time()
            t_iters_total = 0.0  # total time spent in iterations

            print("\nStarting iterations.")
            for iteration in range(n_iterations):

                t_iter_start = time.time()

                batch, text_inputs, document_sizes, sentence_sizes, \
                        author_to_row, author_to_col, papers_by_author_batchind, paper_num_by_author = dataset.next_batch()
                batch_dict = {self.row: batch[:, 0],
                              self.col: batch[:, 1],
                              self.val: batch[:, 2],
                              self.word_model.inputs: text_inputs,
                              self.word_model.sequence_lengths: sentence_sizes,
                              self.sentence_model.sequence_lengths: document_sizes,
                              self.author_to_row: author_to_row,
                              self.author_to_col: author_to_col,
                              self.papers_by_author_batchind: papers_by_author_batchind,
                              self.paper_num_by_author: paper_num_by_author
                              }

                # alternate between optimizing inputs, nnet weights, document model params, and word embeddings
                sess.run(train_lvars, feed_dict=batch_dict)
                sess.run(train_nnet, feed_dict=batch_dict)
                sess.run(train_embeddings, feed_dict=batch_dict)

                iter_time = time.time() - t_iter_start
                t_iters_total += iter_time

                if iteration % 20 == 0:

                    print(iteration, end="")

                    if not no_train_metric:
                        batch, text_inputs, document_sizes, sentence_sizes, \
                                author_to_row, author_to_col, papers_by_author_batchind, paper_num_by_author = dataset.get_training_set()
                        train_dict = {self.row: batch[:, 0],
                                      self.col: batch[:, 1],
                                      self.val: batch[:, 2],
                                      self.word_model.inputs: text_inputs,
                                      self.word_model.sequence_lengths: sentence_sizes,
                                      self.sentence_model.sequence_lengths: document_sizes,
                                      self.author_to_row: author_to_row,
                                      self.author_to_col: author_to_col,
                                      self.papers_by_author_batchind: papers_by_author_batchind,
                                      self.paper_num_by_author: paper_num_by_author
                                      }

                        train_loss_ = sess.run(self.loss, feed_dict=train_dict)
                        train_loss_summary_str = sess.run(train_loss_summary, feed_dict={train_loss: train_loss_})
                        writer.add_summary(train_loss_summary_str, iteration)
                        print("\tTrain loss: %.4f" % train_loss_, end="")


                    if holdout_ratio is not None:
                        batch, text_inputs, document_sizes, sentence_sizes, \
                                author_to_row, author_to_col, papers_by_author_batchind, paper_num_by_author = dataset.get_testing_set()
                        test_dict = {self.row: batch[:, 0],
                                     self.col: batch[:, 1],
                                     self.val: batch[:, 2],
                                     self.word_model.inputs: text_inputs,
                                     self.word_model.sequence_lengths: sentence_sizes,
                                     self.sentence_model.sequence_lengths: document_sizes,
                                     self.author_to_row: author_to_row,
                                     self.author_to_col: author_to_col,
                                     self.papers_by_author_batchind: papers_by_author_batchind,
                                     self.paper_num_by_author: paper_num_by_author
                                     }

                        test_xent_ = sess.run(self.entropy, feed_dict=test_dict)
                        test_xent_summary_str = sess.run(test_xent_summary, feed_dict={test_xent: test_xent_})
                        writer.add_summary(test_xent_summary_str, iteration)
                        print("\tTest xent: %.4f" % test_xent_, end="")

                    total_mins = (time.time() - t_start) / 60.0
                    ave_per_iter = t_iters_total / (iteration + 1.)
                    print("\tTot. time: %.2f mins (ave. %.2f secs/iter)" % (total_mins, ave_per_iter))

            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()




if __name__=='__main__':

    data_file = "/Users/koa/github-repos/bayes-nnet-mf/data/nips-static-small.pkl"

    if not os.path.exists(data_file):
        print("Did not find a stored static dataset. Creating one...")
        data_handler.make_static_dataset(data_file, small=True)

    # load the formatted dataset
    print("Loading and formatting stored dataset...")
    adj_matrix, documents, vocabulary, papers_by_author, paper_authors_df, authors_df \
        = data_handler.load_and_format_data(data_file)

    m = NipsHanNetwork(word_embedding_size=12, word_encoder_size=11, word_attention_size=10,
                       sentence_encoder_size=9, sentence_attention_size=8,
                       n_factors=7, d_pairwise=6, hidden_layer_sizes=[5, 4])

    root_savedir = "/Users/Koa/github-repos/bayes-nnet-mf/saved/nips-han/"
    root_logdir = os.path.join(root_savedir, "tf_logs")

    m.train(adj_matrix, documents, papers_by_author,
            n_iterations=3, batch_size=100, learning_rate=0.01,
            reg_param=0.01, l2_param=0.01,
            root_savedir=root_savedir, root_logdir=root_logdir,
            holdout_ratio=0.03, no_train_metric=True,
            seed=None)