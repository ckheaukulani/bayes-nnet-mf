import os
import sys
import pickle
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

sys.path.extend(['/Users/Koa/github-repos/han-core'
                ])
from eager.sentence_level import SentenceLevel
from eager.word_level import WordLevel

import data_handler


# Turn on eager execution in Tensorflow
tf.enable_eager_execution()
print("Tensorflow eagerly executing?", tf.executing_eagerly())


class NipsHanNetwork(tf.keras.Model):
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
        super().__init__()

        self.word_embedding_size = word_embedding_size
        self.word_encoder_size = word_encoder_size
        self.word_attention_size = word_attention_size
        self.sentence_encoder_size = sentence_encoder_size
        self.sentence_attention_size = sentence_attention_size

        self.n_factors = n_factors
        self.d_pairwise = d_pairwise
        self.hidden_layer_sizes = hidden_layer_sizes


    def loss(self, batch, text_inputs, document_sizes, sentence_sizes, row_doc_inds, col_doc_inds):
        """
        The forward pass of the model.

        :param batch:
        :param text_inputs:
        :param document_sizes: (batch_size,)
        :param sentence_sizes: (batch_size, max_document_size)
        :param row_doc_inds:
        :param col_doc_inds:
        :return:
        """

        sentence_vectors, _ = self.word_model.get_sentence_vectors(inputs=text_inputs,
                                                                   sequence_lengths=sentence_sizes)

        document_vectors, _ = self.sentence_model.get_document_vectors(sentence_vectors=sentence_vectors,
                                                                       sequence_lengths=document_sizes)  # (doc-batch_size, embedding_size)

        row = batch[:, 0]
        col = batch[:, 1]
        val = batch[:, 2]

        # For every batch element, we need the list of 0-indices into 'document_vectors' that needs to be collected and
        # summed.
        row_doc_inds = [np.array(x, dtype=int) for x in row_doc_inds]
        col_doc_inds = [np.array(x, dtype=int) for x in col_doc_inds]

        # This is nicely written as one line, but since it's a loop over the batch it will probably be a bottleneck.
        # Attempted to use tf.map_fun, but it appears 'elems' (to be iterated over) must be a rectangular array.
        row_embeddings = tf.stack([tf.reduce_sum(tf.gather(document_vectors, indices=inds_), axis=0)
                                   for inds_ in row_doc_inds],
                                  axis=0)

        col_embeddings = tf.stack([tf.reduce_sum(tf.gather(document_vectors, indices=inds_), axis=0)
                                   for inds_ in col_doc_inds],
                                  axis=0)



        ###  BUILD THE NETWORK MODEL  ###

        with tf.variable_scope("matrix_factorization"):

            inputs_ = tf.concat([tf.gather(self.U, indices=row),
                                 tf.gather(self.U, indices=col),
                                 tf.gather(self.Up, indices=row) * tf.gather(self.Up, indices=col),
                                 row_embeddings,
                                 col_embeddings
                                 ], axis=-1)  # (edge-batch_size, n_inputs)

            activation_fn = tf.nn.relu

            weights_regularizer = tf.contrib.layers.l2_regularizer(self.l2_param) if self.l2_param is not None else None

            for layer_size in self.hidden_layer_sizes:
                inputs_ = tf.layers.dense(inputs_, layer_size, activation=activation_fn,
                                          kernel_regularizer=weights_regularizer)

            # output layer
            logits = tf.layers.dense(inputs_, 1, activation=None, kernel_regularizer=weights_regularizer,
                                     name="output_layer")

        entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(val, tf.float32),
                                                                        logits=tf.squeeze(logits)))

        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if self.l2_param is not None else []
        reg_losses = tf.losses.get_regularization_losses() if self.l2_param is not None else []
        print("\nReg losses:", reg_losses)

        # FIXME: tf.losses.get_regularization_losses() does not appear to correctly fetch losses in Eager mode

        loss = entropy \
                + self.reg_param * (tf.reduce_sum(tf.square(self.U))
                              + tf.reduce_sum(tf.square(self.Up))
                              )

        loss = tf.add_n([loss] + reg_losses, name='loss')

        return loss, entropy


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

        self.l2_param = l2_param
        self.reg_param = reg_param

        self.dataset = data_handler.Dataset(adj_matrix, documents, papers_by_author,
                                            batch_size=batch_size, holdout_ratio=holdout_ratio, seed=seed)

        vocabulary_size = self.dataset.vocabulary_size

        ###  BUILD THE DOCUMENT MODEL  ###

        # first create the word level model, which also creates placeholders for data
        self.word_model = WordLevel(vocabulary_size=vocabulary_size,
                                    word_embedding_size=self.word_embedding_size,
                                    word_encoder_size=self.word_encoder_size,
                                    word_attention_size=self.word_attention_size
                                    )

        # the sentence vectors from the word model are passed into the sentence level model
        self.sentence_model = SentenceLevel(sentence_encoder_size=self.sentence_encoder_size,
                                            sentence_attention_size=self.sentence_attention_size
                                            )

        with tf.variable_scope("matrix_factorization"):
            self.U = tf.Variable(tf.random_normal([self.dataset.n_authors, self.n_factors]), name='U')
            self.Up = tf.Variable(tf.random_normal([self.dataset.n_authors, self.d_pairwise]), name='Up')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


        #
        # print("\nlatent vars:", latent_vars)
        # print("\nnnet vars:", nnet_vars)
        # print("\nword embeddings:", self.word_model.word_embeddings)


        ###  Training  ###

        t_start = time.time()
        t_iters_total = 0.0  # total time spent in iterations

        print("\nStarting iterations.")
        for iteration in range(n_iterations):

            t_iter_start = time.time()

            # grab the batch
            batch, text_inputs, document_sizes, sentence_sizes, row_doc_inds, col_doc_inds = self.dataset.next_batch()

            # to collect up all gradients and variables
            grad_fn = tfe.implicit_gradients(self.loss)
            grads_and_vars = grad_fn(batch, text_inputs, document_sizes, sentence_sizes, row_doc_inds, col_doc_inds)

            # all_vars = [var_ for _, var_ in grads_and_vars]
            # print("\n all vars:", [var_.name for var_ in all_vars])

            # update nnet vars
            latent_vars = [self.U, self.Up]  # the inputs to the nnets
            nnet_grad_and_vars = [(grad_, var_) for grad_, var_ in grads_and_vars
                                  if var_ not in latent_vars + [self.word_model.word_embeddings]]

            optimizer.apply_gradients(nnet_grad_and_vars)

            # update network model inputs
            with tf.GradientTape() as tape:
                loss, _ = self.loss(batch, text_inputs, document_sizes, sentence_sizes, row_doc_inds, col_doc_inds)

            grads_ = tape.gradient(loss, latent_vars)
            optimizer.apply_gradients(zip(grads_, latent_vars))

            # update the word embeddings
            with tf.GradientTape() as tape:
                loss, _ = self.loss(batch, text_inputs, document_sizes, sentence_sizes, row_doc_inds, col_doc_inds)

            grads_ = tape.gradient(loss, [self.word_model.word_embeddings])
            optimizer.apply_gradients(zip(grads_, [self.word_model.word_embeddings]))


            iter_time = time.time() - t_iter_start
            t_iters_total += iter_time

            print(iteration)

            # if iteration % 20 == 0:
            #
            #     print(iteration, end="")
            #
            #     if not no_train_metric:
            #         batch, text_inputs, document_sizes, sentence_sizes, \
            #                 author_to_row, author_to_col, papers_by_author_batchind, paper_num_by_author = dataset.get_training_set()
            #
            #         train_loss_ = sess.run(self.loss, feed_dict=train_dict)
            #
            #         train_loss_summary_str = sess.run(train_loss_summary, feed_dict={train_loss: train_loss_})
            #         writer.add_summary(train_loss_summary_str, iteration)
            #         print("\tTrain loss: %.4f" % train_loss_, end="")
            #
            #
            #     if holdout_ratio is not None:
            #         batch, text_inputs, document_sizes, sentence_sizes, \
            #                 author_to_row, author_to_col, papers_by_author_batchind, paper_num_by_author = dataset.get_testing_set()
            #
            #         test_xent_ = sess.run(self.entropy, feed_dict=test_dict)
            #         test_xent_summary_str = sess.run(test_xent_summary, feed_dict={test_xent: test_xent_})
            #         writer.add_summary(test_xent_summary_str, iteration)
            #         print("\tTest xent: %.4f" % test_xent_, end="")
            #
            #     total_mins = (time.time() - t_start) / 60.0
            #     ave_per_iter = t_iters_total / (iteration + 1.)
            #     print("\tTot. time: %.2f mins (ave. %.2f secs/iter)" % (total_mins, ave_per_iter))




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
            n_iterations=3, batch_size=32, learning_rate=0.01,
            reg_param=0.01, l2_param=0.01,
            root_savedir=root_savedir, root_logdir=root_logdir,
            holdout_ratio=0.03, no_train_metric=True,
            seed=None)