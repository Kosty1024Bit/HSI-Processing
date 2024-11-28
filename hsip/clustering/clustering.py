import numpy as np

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist

from tqdm import tqdm


class CosClust():
    '''
    A clustering algorithm based on cosine similarity. Groups samples into clusters based on a 
    threshold for cosine similarity and assigns labels to each sample.

    Parameters
    ----------
    threshold : float, optional
        The cosine similarity threshold for determining cluster membership. Default is 0.9.
    verbose : bool, optional
        If True, displays progress and additional information during the clustering process. Default is True.

    Attributes
    ----------
    threshold : float
        The cosine similarity threshold for clustering.
    labels : np.ndarray or None
        The cluster labels assigned to each sample. Initialized as `None` and populated after `fit` is called.
    reference_set : list
        A list of reference samples representing each cluster.

    Methods
    -------
    fit(source_data, y=None)
        Performs clustering on the input data and returns cluster labels.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 50)  # 100 samples with 50 features each
    >>> model = CosClust(threshold=0.8, verbose=True)
    >>> labels = model.fit(data)
    >>> print(labels)
    array([0, 1, 0, 2, ..., 1])  # Example output
    '''
        
    def __init__(self, threshold: float = 0.9, verbose=True):
        self.threshold = threshold
        self.labels = None

        self.reference_set = []


    def fit(self, source_data: np.ndarray, y=None):
        '''
        Performs clustering on the given data based on cosine similarity.

        Parameters
        ----------
        source_data : np.ndarray
            A 2D array of shape (n_samples, n_features) where `n_samples` is the number of samples 
            and `n_features` is the number of features for each sample.
        y : Ignored
            Placeholder for compatibility, not used in this implementation.

        Returns
        -------
        np.ndarray
            A 1D array of shape (n_samples,) containing the cluster labels for each sample.

        Notes
        -----
        - The algorithm proceeds in two steps:
            1. Initial clustering based on the cosine similarity threshold.
            2. Adjustment of labels based on the reference set of cluster representatives.
        - Samples with no cluster assignment are labeled as `-1`.
        '''
        
        n_clust = 0
        self.labels = np.full(shape=source_data.shape[0], fill_value=-1, dtype=int)
        id_reference_set = []

        n_samples, _ = source_data.shape
        
        if verbose:
            print("CosClust take a step 1/2")
        for i in tqdm(range(n_samples), disable=not verbose):
            if self.labels[i] == -1:
                corr_array = 1 - cdist(source_data[i, np.newaxis], source_data, "cosine")[0]

                bool_mask = corr_array > self.threshold
                bool_mask[self.labels != -1] = False
                self.labels[bool_mask] = n_clust

                n_clust += 1

                id_reference_set.append(i)
                self.reference_set.append(source_data[i])

        tmp_labels = np.full(shape=source_data.shape[0], fill_value=-1, dtype=int)
        
        if verbose:
            print("CosClust take a step 2/2")
        for i in tqdm(range(n_samples), disable=not verbose):
            tmp_corr_array = 1 - cdist(source_data[i, np.newaxis], source_data[id_reference_set], "cosine")[0]
            if id_reference_set[self.labels[i]] == i:
                tmp_corr_array[self.labels[i]] = 0.0

            tmp_labels[i] = tmp_corr_array.argmax()

        self.labels = tmp_labels

        return self.labels