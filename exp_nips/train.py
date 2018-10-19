import numpy as np
import scipy


def load_data():

    mat = scipy.io.load_mat(os.path.join(root, 'data', 'nips', 'nips_1-17.mat'))

    authors_names = mat['authors_names']
    authors_names = np.concatenate((np.array(authors_names)[0, :]))

    X = mat['docs_authors'].copy()
    X = np.dot(X.T, X)
    X = (X>0).astype(int)
    X = X.toarray()

    to_keep = np.nonzero(np.sum(X, axis=0) > 6)[0]
    X = X[to_keep, :]
    X = X[:, to_keep]
    authors_names = authors_names[to_keep]
    
    n_users = X.shape[1]
    il = np.tril_indices(X.shape[0], k=-1)
    vals = np.array(X[il]).ravel()
    data = np.vstack([il[0], il[1], vals]).T

    return data, authors_names
