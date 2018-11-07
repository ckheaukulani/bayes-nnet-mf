import os
import sys
import pickle
import numpy as np
import tensorflow as tf

sys.path.extend(['/Users/Koa/github-repos/bayes-nnet-mf',
                 '/Users/Koa/github-repos/han-core'
                ])
from sentence_level import SentenceLevel
from word_level import WordLevel

import data_handler
import utils

class HAN_NIPS:
    def __init__(self, word_embedding_size, word_encoder_size, word_attention_size,
                 sentence_encoder_size, sentence_attention_size):
        """
        Main class for the hierarchical attention network model for document embeddings.
        """
        self.word_embedding_size = word_embedding_size
        self.word_encoder_size = word_encoder_size
        self.word_attention_size = word_attention_size
        self.sentence_encoder_size = sentence_encoder_size
        self.sentence_attention_size = sentence_attention_size

    def construct_graph(self):

        # first create the word level model, which also creates placeholders for data
        self.word_model = WordLevel(vocabulary_size=self.vocabulary_size,
                                    word_embedding_size=self.word_embedding_size,
                                    word_encoder_size=self.word_encoder_size,
                                    word_attention_size=self.word_attention_size
                                    )

        # the sentence vectors from the word model are passed into the sentence level model
        self.sentence_model = SentenceLevel(sentence_vectors=self.word_model.sentence_vectors,
                                            sentence_encoder_size=self.sentence_encoder_size,
                                            sentence_attention_size=self.sentence_attention_size
                                            )

        row = tf.placeholder(dtype=tf.int32, shape=[None], name='row')
        col = tf.placeholder(dtype=tf.int32, shape=[None], name='col')
        val = tf.placeholder(dtype=tf.int32, shape=[None], name='val')

        """ HERE! Difficulty collecting up document vectors by author """

        # map the constructed document vectors to become features for their authors; each author's feature is a sum of
        # their corresponding document vectors
        self.papers_by_author = tf.placeholder(dtype=tf.int32,
                                               shape=[self.n_authors, self.n_max_papers],
                                               name='papers_by_author')
            # the integer indices of the papers belonging to each author IN THIS BATCH (so many rows will be all NaNs)

        self.authors_batch = tf.placeholder(dtype=tf.int32, shape=[None], name='authors_batch')  # size of document-batch
            # the author IDs corresponding to the documents in the document-batch

        row_papers_by_author = tf.gather(self.papers_by_author, indices=row, axis=0)  # (edge-batch_size, n_max_papers)


        doc_to_batch_inds = {}

        # the constructed document vectors will be of shape (n_docs_in_batch, 2 * sentence_encoder_size)
        for row, col in
        tf.gather(self.sentence_model.document_vectors, indices=, axis=0)

        # The document vectors now summarize the documents. For now, concatenate them with a simple sum for every author.



    def train(self, adj_matrix, documents, paper_list_by_authors,
              n_iterations, batch_size, learning_rate,
              root_savedir, root_logdir,
              holdout_ratio=None, no_train_metric=False,
              seed=None):
        """
        Training routine.

        :param adj_matrix:
        :param documents:
        :param paper_list_by_authors:
        :param n_iterations:
        :param batch_size:
        :param root_savedir:
        :param root_logdir:
        :param learning_rate:
        :param no_train_metric:
        :param seed:
        :return:
        """

        self.n_authors = adj_matrix['n_authors']
        rows = adj_matrix['row']
        cols = adj_matrix['col']

        self.vocabulary_size = vocabulary_size
        self.n_max_papers = n_max_papers


        # create the batch generator
        pairs = utils.get_pairs(self.n_authors, rows, cols)
        pairs = pairs.astype(int)
        batch_generator = data_handler.BatchGenerator(pairs, documents, batch_size=batch_size, holdout_ratio=holdout_ratio, seed=seed)


        # keep track of the global step when storing the TF session
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # training flag for batch normalization and dropout
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')


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

        writer = tf.summary.FileWriter(root_logdir)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            init.run()

            for iteration in range(n_iterations):

                batch, docs_in_batch, text_inputs, document_sizes, sentence_sizes = batch_generator.next_batch()
                batch_dict = {self.row: batch[:, 0],
                              self.col: batch[:, 1],
                              self.val: batch[:, 2],
                              self.inputs: text_inputs,
                              self.word_model.sequence_lengths: sentence_sizes,
                              self.sentence_model.sequence_lengths: document_sizes
                              }

                # alternate between optimizing inputs, nnet weights, document model params, and word embeddings
                sess.run(train_lvars, feed_dict=batch_dict)
                sess.run(train_nnet, feed_dict=batch_dict)

                if iteration % 20 == 0:

                    print(iteration, end="")

                    if not no_train_metric:
                        train_dict = {self.row: batch_generator.train[:, 0],
                                      self.col: batch_generator.train[:, 1],
                                      self.val: batch_generator.train[:, 2]}

                        train_loss_ = sess.run(self.loss, feed_dict=train_dict)
                        train_loss_summary_str = sess.run(train_loss_summary, feed_dict={train_loss: train_loss_})
                        writer.add_summary(train_loss_summary_str, iteration)
                        print("\ttrain loss: %.4f" % train_loss_, end="")


                    if holdout_ratio is not None:
                        test_dict = {self.row: batch_generator.test[:, 0],
                                     self.col: batch_generator.test[:, 1],
                                     self.val: batch_generator.test[:, 2]}

                        test_xent_ = sess.run(self.entropy, feed_dict=test_dict)
                        test_xent_summary_str = sess.run(test_xent_summary, feed_dict={test_xent: test_xent_})
                        writer.add_summary(test_xent_summary_str, iteration)
                        print("\ttest xent: %.4f" % test_xent_)


            # save the model
            saver.save(sess, os.path.join(root_savedir, "model.ckpt"))

        # close the file writer
        writer.close()


if __name__=='__main__':

    data_file = "/Users/koa/github-repos/bayes-nnet-mf/data/nips-static.pkl"

    if not os.path.exists(data_file):
        data_root = "/Users/koa/datasets/nips-papers/"
        data_handler.make_static_dataset(data_root, data_file)

    # load the formatted dataset
    adj_matrix, documents, paper_list_by_authors, paper_authors_df, authors_df = data_handler.load_and_format_data(data_file)