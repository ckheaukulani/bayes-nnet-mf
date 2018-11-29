import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

sys.path.extend(['/Users/Koa/github-repos/dynamic-nnet-rm',
                 '/Users/Koa/github-repos/bayes-nnet-mf'
                ])
import get_nips_data, utils


def make_static_dataset(savename, small=False):
    """
    Wrapper to make a static dataset, consisting of an adjacency matrix and all processed text documents. We have no
    reason to trim the number of authors here.

    :param savename:
    :param small: boolean whether to make a small, toy dataset for dev purposes
    :return:
    """

    # load the data
    if small:
        paper_text_df, paper_authors_df, authors_df = get_nips_data.load_small_data()
        print("Saving a small/dev dataset under the filename", savename)

    else:
        paper_text_df, paper_authors_df, authors_df = get_nips_data.load_data()

        # first trim the dataset of authors without enough co-authorship links
        paper_text_df, paper_authors_df, authors_df = \
            get_nips_data.trim_by_coauthor_links(paper_text_df, paper_authors_df, authors_df, min_threshold=3)

    # process the text data, since in theory this could remove some documents
    documents = get_nips_data.process_text_data(paper_text_df)
    del paper_text_df

    # The format of 'documents' is a list like
    # [{'id': <id>, 'year': <year>, 'title': <title>, 'sentences': [[<str>, ...], ..., [<str>, ...]]}, ...]

    # make the co-authorship count matrix (adjacency matrix with counts)
    counts, author_id_map = get_nips_data.make_coauthor_counts(paper_authors_df)  # returns a scipy.sparse.csr_matrix
    counts = counts.tocoo()  # conversion is fast
    adj_matrix = {'n_authors': counts.shape[0],
                  'row': counts.row,
                  'col': counts.col,
                  'author_id_map': author_id_map  # bind this to the adjacency matrix data
                  }

    # we need no info other than the text, so reformat the documents data to a dictionary like
    # {paper_id: <list of lists corresponding to sentences>}
    documents = {paper['id']: paper['sentences'] for paper in documents}

    # convert the word tokens to integers
    print("Converting word tokens to integers...")
    documents, vocabulary = convert_tokens_to_ints(documents)

    # bind all this data together and save it
    with open(savename, "wb") as f:
        pickle.dump((adj_matrix, documents, vocabulary, paper_authors_df, authors_df), f)


def load_and_format_data(data_file):
    """
    Load the NIPS papers dataset, referring to 'make_static_dataset()' in 'get_nips_data.py' to understand how the data
    was processed. Loading the dataset gives:
        - counts_matrix: a dictionary with entries 'n_authors', 'row', 'col', 'val' containing the coauthorship COUNTS.
        - documents: a dictionary of the format {paper_id: <list of lists corresponding to sentences>}, where each
            sentence is a list of word tokens.
        - paper_authors_df: Pandas dataframe with columns 'paper_id' and 'author_id', which can be used to 0-index arrays.
        - authors_df: Pandas dataframe with columns 'id' and 'name'.
    :return:
    """

    with open(data_file, "rb") as f:
        adj_matrix, documents, vocabulary, paper_authors_df, authors_df = pickle.load(f)

    # for each author, make a list of indices pointing to the documents by that author
    papers_by_authors = {author_id: [] for author_id in authors_df['id'].unique()}
    for ind_ in paper_authors_df.index:
        author_id = paper_authors_df.loc[ind_, 'author_id']
        paper_id = paper_authors_df.loc[ind_, 'paper_id']
        papers_by_authors[author_id].append(paper_id)

    ###  Print info about the loaded dataset  ###
    print("\nLoading and formatting dataset complete.")

    n_authors = adj_matrix['n_authors']
    n_papers = len(documents.keys())
    print("There are {} authors and {} papers.".format(n_authors, n_papers))

    n_links = len(adj_matrix['row'])
    density = n_links / (n_links ** 2.0)
    print("The co-authorship matrix has %d links (density: %.2f)" % (n_links, density))

    n_obs = sum([len(s_) for _, d_ in documents.items() for s_ in d_])
    vocab_size = len(vocabulary.keys())
    print("The text contains {} observations from {} vocabulary terms.".format(n_obs, vocab_size))

    return adj_matrix, documents, vocabulary, papers_by_authors, paper_authors_df, authors_df


def convert_tokens_to_ints(documents):
    """
    Convert the word tokens into integers.

    :param documents: a dictionary of the format {paper_id: <list of lists corresponding to sentences>}, where each
            sentence is a list of word tokens.
    :return: documents, vocabulary
        - documents: Takes the exact same format except with word (string) tokens replaced by integer tokens
        - vocabulary: Dict {'word': token} where 'word' is the original word token, and 'token' is the new corresponding
            integer token.
    """

    unique_words = list(set([w for _, s_ in documents.items() for l_ in s_ for w in l_]))

    for doc_id in documents.keys():
        int_doc = []
        for sentence in documents[doc_id]:
            int_sentence = []
            for w in sentence:
                i_ = unique_words.index(w)
                int_sentence.append(i_)
            int_doc.append(int_sentence)
        documents[doc_id] = int_doc

    # return the word mapping... to be useful, it should take the integer ID and return the token
    vocabulary = {i_: w for i_, w in enumerate(unique_words)}

    return documents, vocabulary


class Dataset:
    def __init__(self, adj_matrix, documents, papers_by_author,
                 batch_size=None, holdout_ratio=None, seed=None):
        """
        Considering what a minibatch looks like here: its elements correspond to a tuple (i, j) of nodes in the graph.

        :param n_authors:
        :param rows:
        :param cols:
        :param author_id_map:
        :param documents:
        :param papers_by_author: dict like {<author_id>: [<paper_id>, ...], ...}
        :param batch_size:
        :param holdout_ratio:
        :param seed:
        """

        np.random.seed(seed)

        # required entries in the dict 'adj_matrix'
        n_authors = adj_matrix['n_authors']
        author_id_map = adj_matrix['author_id_map']

        rows = adj_matrix['row']
        cols = adj_matrix['col']

        self.n_authors = n_authors

        # collect up all pairs in the binary matrix and format as a single array
        pairs = utils.get_pairs(n_authors, rows, cols)
        pairs = pairs.astype(int)

        # remove missing indices, if any
        # TODO: Test this
        if 'miss_row' in list(adj_matrix.keys()):

            miss_rows = adj_matrix['miss_row']
            miss_cols = adj_matrix['miss_col']

            assert len(np.unique(np.concatenate([rows, miss_rows]))) == (len(rows) + len(miss_rows))  # should not overlap
            assert len(np.unique(np.concatenate([cols, miss_cols]))) == (len(cols) + len(miss_cols))  # should not overlap

            # searching the indices sequentially would be too slow, so dump into a dictionary
            ind_map = {(i_, j_): ind_ for ind_, (i_, j_) in enumerate(zip(pairs[:, 0], pairs[:, 1]))}
            miss_inds = [ind_map[(i_, j_)] for i_, j_ in zip(miss_rows, miss_cols)]

            pairs = np.delete(pairs, miss_inds, axis=0)
            print("Removed {} missing entries, resulting in {} edges.".format(len(miss_inds), len(pairs)))

        if batch_size is None:
            batch_size = len(pairs)
            print("No batch size. Using batch learning.")

        holdout_ratio = None if holdout_ratio == 0 else holdout_ratio

        # process the pairs into train/test sets and create batch generator; after this, 'pairs' will no longer be referenced
        self._process_pairs(pairs, holdout_ratio, batch_size)

        # re-associate authors and documents with their 0-indices
        self.author_id_map = author_id_map  # the mapping from author ID to the 0-index in the adjacency matrix; like {'author_id': <0-index>}

        # similarly we also need a map from doc IDs to 0-indices
        unique_doc_ids = list(documents.keys())
        assert len(unique_doc_ids) == len(set(unique_doc_ids))
        self.paper_id_map = {id_: ind_ for ind_, id_ in enumerate(unique_doc_ids)}

        # remap the indices IN 'documents' and 'papers_by_author'
        self.documents = {self.paper_id_map[doc_id]: doc for doc_id, doc in documents.items()}
        self.papers_by_author = {self.author_id_map[author_id]: [self.paper_id_map[paper_id] for paper_id in paper_list]
                                 for author_id, paper_list in papers_by_author.items()}

        self.batch_size = batch_size
        self.holdout_ratio = holdout_ratio

        self.vocabulary_size = len(set([w_ for _, d_ in documents.items() for s_ in d_ for w_ in s_]))
        self.max_num_docs = max([len(l_) for _, l_ in papers_by_author.items()])  # a required constant

    def _process_pairs(self, pairs, holdout_ratio, batch_size):
        """
        Create the edge-batch generator object. An edge-minibatch is defined by a subset of edges in the adjacency
        matrix, i.e., subsets of rows in 'pairs'.

        :param pairs:
        :param holdout_ratio:
        :param batch_size:
        :return:
        """
        num_train = int((1.0 - holdout_ratio) * len(pairs)) if holdout_ratio is not None else len(pairs)
        np.random.shuffle(pairs)

        self.train = pairs[:num_train]
        self.test = pairs[num_train:] if holdout_ratio is not None else None

        self.batch_size = batch_size
        self.num_train_batches = int(np.ceil(len(self.train) / batch_size))  # final batch may be incomplete
        self.num_test_batches = int(np.ceil(len(self.test) / batch_size)) if holdout_ratio is not None else None

        self.train_bind = 0
        self.train_idx = list(range(len(self.train)))
        self.test_bind = 0 if holdout_ratio is not None else None

        if holdout_ratio is not None:
            print("Train/test split: {}/{}".format(len(self.train), len(self.test)))

        print("Batch size {} results in {} training batches.".format(batch_size, self.num_train_batches))

    def _get_idx(self):
        return self.train_idx[self.train_bind * self.batch_size:\
                (self.train_bind + 1) * self.batch_size]

    def _incr_bind(self):
        self.train_bind += 1
        if (self.train_bind == self.num_train_batches):  # final batch is taken care of, slices may overflow an array
            self.train_bind = 0
            np.random.shuffle(self.train_idx)

    def _process_batch(self, batch):
        """
        An edge-batch contains the minibatch of edges (edge-minibatch) in the adjacency matrix; there is a corresponding
        minibatch of documents, which is decided based on the authors in the edge-minibatch.

        :param batch: The edge-minibatch, represented as a (row, col, val) array of size (edge-batch_size, 3)
        :return:
        """

        # find the authors in the edge-batch (note that authors are represented here by their 0-indices)
        authors_in_batch = np.unique(np.concatenate([batch[:, 0], batch[:, 1]]))

        # pull out the "batch" of documents
        papers_by_author_batch = [self.papers_by_author[author_ind] for author_ind in authors_in_batch]  # list of lists
        docs_in_batch = np.unique([x for l_ in papers_by_author_batch for x in l_])  # flattened list of unique doc IDs

        # form the inputs to the RNN, which takes the format of a list of lists of lists; the first level corresponds to
        # documents, the second to sentences, and the base lists are lists of word tokens (as integer IDs)
        text_inputs = [self.documents[paper_ind] for paper_ind in docs_in_batch]

        # format the batch for use with Tensorflow's dynamic RNN functions (largely entails padding)
        text_inputs, document_sizes, sentence_sizes = self._format_text_batch(text_inputs)

        # for every edge-batch entry, provide the 0-indices into 'docs_in_batch' for the papers corresponding to that author
        doc_id_to_ind = {id_: ind_ for ind_, id_ in enumerate(docs_in_batch)}  # for speed

        row_doc_inds = np.zeros([len(batch), self.max_num_docs])
        for i_, author_ in enumerate(batch[:, 0]):
            for j_, id_ in enumerate(self.papers_by_author[author_]):
                row_doc_inds[i_, j_] = doc_id_to_ind[id_]

        col_doc_inds = np.zeros([len(batch), self.max_num_docs])
        for i_, author_ in enumerate(batch[:, 1]):
            for j_, id_ in enumerate(self.papers_by_author[author_]):
                col_doc_inds[i_, j_] = doc_id_to_ind[id_]

        row_doc_nums = np.array([len(self.papers_by_author[i_]) for i_ in batch[:, 0]], dtype=int)  # (batch_size,)
        col_doc_nums = np.array([len(self.papers_by_author[i_]) for i_ in batch[:, 1]], dtype=int)  # (batch_size,)

        return batch, text_inputs, document_sizes, sentence_sizes, \
               row_doc_inds, col_doc_inds, row_doc_nums, col_doc_nums


    def _format_text_batch(self, inputs):
        """
        For every minibatch, we need an array of the word integer IDs, padded out to fill up to the max sizes of the
        tensors (at the word and sentence levels). The dynamic RNN methods also require the actual sentence and
        document lengths.

        :param inputs: list of lists of lists (documents, sentences, words); elements of the lists are integer word IDs
        :return:
        """
        batch_size = len(inputs)  # number of documents
        document_sizes = [len(doc) for doc in inputs]  # num sentences in each doc
        max_document_size = max(document_sizes)  # max document length (in number of sentences)

        sentence_sizes_ = [[len(sentence) for sentence in doc] for doc in inputs]
        max_sentence_size = max([max(list_) for list_ in sentence_sizes_])

        # pad the elements of the tensor to fill up to the max sizes; all the loops are unfortunate...
        batch = np.zeros([batch_size, max_document_size, max_sentence_size])
        sentence_sizes = np.zeros([batch_size, max_document_size])
        for i, document in enumerate(inputs):
            for j, sentence in enumerate(document):
                sentence_sizes[i, j] = sentence_sizes_[i][j]
                for k, word in enumerate(sentence):
                    batch[i, j, k] = word

        return batch.astype(int), np.array(document_sizes, dtype=int), sentence_sizes

    def next_batch(self):
        idx = self._get_idx()
        self._incr_bind()
        batch = self.train[idx]
        return self._process_batch(batch)

    # def get_training_set(self):
    #     return self._process_batch(self.train)
    #
    # def get_testing_set(self):
    #     return self._process_batch(self.test)