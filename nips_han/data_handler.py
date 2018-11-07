import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append('/Users/Koa/github-repos/dynamic-nnet-rm')
import get_nips_data


def make_static_dataset(data_root, root_savedir):
    """
    Wrapper to make a static dataset, consisting of an adjacency matrix and all processed text documents. We have no
    reason to trim the number of authors here.

    :return:
    """

    # load the data
    paper_text_df, paper_authors_df, authors_df = get_nips_data.load_data(data_root)

    # first process the text data, since in theory this could remove some documents
    documents = get_nips_data.process_text_data(paper_text_df)
    del paper_text_df

    # making a adjaceny matrix doesn't make sense if the author IDs do not 0-index an array, so remap the author IDs
    authors_df, paper_authors_df = get_nips_data.remap_author_ids(authors_df, paper_authors_df)

    # make the co-authorship count matrix (adjacency matrix with counts)
    counts = get_nips_data.make_coauthor_counts(paper_authors_df)  # returns a scipy.sparse.csr_matrix
    counts = counts.tocoo()  # conversion is fast
    adj_matrix = {'n_authors': counts.shape[0],
                  'row': counts.row,
                  'col': counts.col
                  }

    # this application also needs the document IDs to 0-index arrays
    documents, paper_authors_df = get_nips_data.remap_paper_ids(documents, paper_authors_df)

    # we need no info other than the text, so reformat the documents data to a dictionary like
    # {paper_id: <list of lists corresponding to sentences>}
    documents = {paper['id']: paper['text'] for paper in documents}

    # bind all this data together
    with open(os.path.join(root_savedir, "nips-static.pkl"), "wb") as f:
        pickle.dump((adj_matrix, documents, paper_authors_df, authors_df), f)


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
        adj_matrix, documents, paper_authors_df, authors_df = pickle.load(f)

    # convert the word tokens to integers
    documents, vocabulary = convert_tokens_to_ints(documents)

    # for each author, make a list of indices pointing to the documents
    paper_list_by_authors = {author_id: [] for author_id in authors_df['id'].unique()}
    for ind_ in paper_authors_df.index:
        author_id = paper_authors_df.loc[ind_, 'author_id']
        paper_id = paper_authors_df.loc[ind_, 'paper_id']
        paper_list_by_authors[author_id].extend(paper_id)

    return adj_matrix, documents, paper_list_by_authors, paper_authors_df, authors_df


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

    unique_words = set([w for _, l_ in documents.items() for w in l_])

    for doc in documents:
        int_doc = []
        for sentence in doc['sentences']:
            int_sentence = []
            for w in sentence:
                i_ = unique_words.index(w)
                int_sentence.append(i_)
            int_doc.append(int_sentence)
        doc['sentences'] = int_doc

    # return the word mapping
    vocabulary = {w: i_ for i_, w in enumerate(unique_words)}

    return documents, vocabulary


class BatchGenerator:
    def __init__(self, data, documents, paper_list_by_authors,
                 batch_size=None, holdout_ratio=None, seed=None):
        """
        Considering what a minibatch looks like here: its elements correspond to a tuple (i, j) of nodes in the graph.

        :param data: The adjacency matrix in (row, col, val) array format; shape (n_obs, 3)
        :param documents:
        :param paper_list_by_authors:
        :param batch_size:
        :param holdout_ratio:
        :param seed:
        """

        np.random.seed(seed)

        # a minibatch is defined by a subset of edges in the adjacency, so make that first, as usual
        if batch_size is None:
            batch_size = len(data)
            print("No batch size. Using batch learning.")

        holdout_ratio = None if holdout_ratio == 0 else holdout_ratio

        num_train = int((1.0 - holdout_ratio) * len(data)) if holdout_ratio is not None else len(data)
        np.random.shuffle(data)

        self.train = data[:num_train]
        self.test = data[num_train:] if holdout_ratio is not None else None

        self.batch_size = batch_size
        self.num_train_batches = int(np.ceil(len(self.train) / batch_size))  # final batch may be incomplete
        self.num_test_batches = int(np.ceil(len(self.test) / batch_size)) if holdout_ratio is not None else None

        self.train_bind = 0
        self.train_idx = list(range(len(self.train)))
        self.test_bind = 0 if holdout_ratio is not None else None

        if holdout_ratio is not None:
            print("Train/test split: {}/{}".format(len(self.train), len(self.test)))

        print("Batch size {} results in {} training batches.".format(batch_size, self.num_train_batches))

        # additionally, we will return a list of documents for every author in a minibatch; note there is no concept of
        # 'train' and 'test' documents
        self.documents = documents
        self.paper_list_by_authors = paper_list_by_authors

    def _get_idx(self):
        return self.train_idx[self.train_bind * self.batch_size:\
                (self.train_bind + 1) * self.batch_size]

    def _incr_bind(self):
        self.train_bind += 1
        if (self.train_bind == self.num_train_batches):  # final batch is taken care of, slices may overflow an array
            self.train_bind = 0
            np.random.shuffle(self.train_idx)

    def next_batch(self):
        idx = self._get_idx()
        self._incr_bind()
        batch = self.train[idx]

        # 'batch' contains the minibatch of edges in the adjacency matrix; there is a corresponding minibatch of
        # documents, which is decided based on the authors in the edge-minibatch
        authors_in_batch = np.unique(np.concatenate([batch[:, 0], batch[:, 1]]))

        # pull out the "batch" of documents
        # docs_in_batch = {author_id: self.paper_list_by_authors[author_id] for author_id in authors_in_batch}  # dict of lists
        docs_in_batch = [self.paper_list_by_authors[author_id] for author_id in authors_in_batch]  # list of lists
        docs_in_batch = np.unique([x for l_ in docs_in_batch for x in l_])  # flattened list of unique doc IDs

        # form the inputs to the RNN, which takes the format of a list of lists of lists; the first level corresponds to
        # documents, the second to sentences, and the base lists are lists of word tokens (as integer IDs)
        text_inputs = [self.documents[paper_id] for paper_id in docs_in_batch]

        # format the batch for use with Tensorflow's dynamic RNN functions (largely entails padding)
        text_inputs, document_sizes, sentence_sizes = self._format_text_batch(text_inputs)

        return batch, docs_in_batch, text_inputs, document_sizes, sentence_sizes

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
        max_document_size = document_sizes.max()  # max document length (in number of sentences)

        sentence_sizes_ = [[len(sentence) for sentence in doc] for doc in inputs]
        max_sentence_size = max([max(list_) for list_ in sentence_sizes_])
        # max_sentence_size = max(map(max, sentence_sizes_))

        # pad the elements of the tensor to fill up to the max sizes; all the loops are unfortunate...
        batch = np.zeros([batch_size, max_document_size, max_sentence_size])
        sentence_sizes = np.zeros([batch_size, max_document_size])
        for i, document in enumerate(inputs):
            for j, sentence in enumerate(document):
                sentence_sizes[i, j] = sentence_sizes_[i][j]
                for k, word in enumerate(sentence):
                    batch[i, j, k] = word

        return batch, document_sizes, sentence_sizes