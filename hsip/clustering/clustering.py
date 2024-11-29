import numpy as np

from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch

from hdbscan import HDBSCAN as source_HDBSCAN

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
        self.verbose = verbose
        self.reference_set = []


    def fit(self, source_data: np.ndarray):
        '''
        Performs clustering on the given data based on cosine similarity.

        Parameters
        ----------
        source_data : np.ndarray
            A 2D array of shape (n_samples, n_features) where `n_samples` is the number of samples 
            and `n_features` is the number of features for each sample.

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
        
        if source_data.shape[0] > 512000:
            raise ValueError(f'Very large sample! Recommended no more than 512000 samples. Submitted: {source_data.shape[0]}.')

        n_clust = 0
        self.labels = np.full(shape=source_data.shape[0], fill_value=-1, dtype=int)
        id_reference_set = []

        n_samples, _ = source_data.shape
        
        if self.verbose:
            print("CosClust take a step 1/2")
        for i in tqdm(range(n_samples), disable=not self.verbose):
            if self.labels[i] == -1:
                corr_array = 1 - cdist(source_data[i, np.newaxis], source_data, "cosine")[0]

                bool_mask = corr_array > self.threshold
                bool_mask[self.labels != -1] = False
                self.labels[bool_mask] = n_clust

                n_clust += 1

                id_reference_set.append(i)
                self.reference_set.append(source_data[i])

        tmp_labels = np.full(shape=source_data.shape[0], fill_value=-1, dtype=int)
        
        if self.verbose:
            print("CosClust take a step 2/2")
        for i in tqdm(range(n_samples), disable=not self.verbose):
            tmp_corr_array = 1 - cdist(source_data[i, np.newaxis], source_data[id_reference_set], "cosine")[0]
            if id_reference_set[self.labels[i]] == i:
                tmp_corr_array[self.labels[i]] = 0.0

            tmp_labels[i] = tmp_corr_array.argmax()

        self.labels = tmp_labels

        return self.labels


class SCH():
    '''
    A class for performing hierarchical clustering using SciPy's `linkage` and `fcluster` methods.

    The `SCH` class provides an interface for hierarchical clustering with customizable linkage 
    and clustering parameters.

    Attributes
    ----------
    labels : np.ndarray or None
        Cluster labels assigned to each data point after fitting the model. Initially set to `None`.
    linkage_method : str
        Linkage method used for hierarchical clustering. Supported methods include 
        `"single"`, `"complete"`, `"average"`, `"weighted"`, `"centroid"`, `"median"`, and `"ward"`.
    linkage_metric : str
        Distance metric used to compute pairwise distances between data points. Common metrics 
        include `"euclidean"`, `"cosine"`, `"cityblock"`, and `"hamming"`.
    linkage_optimal_ordering : bool
        If `True`, the linkage matrix will be reordered to minimize the distances between successive leaves.
    fcluster_t : float
        The threshold to apply when forming flat clusters. The meaning of `t` depends on the 
        `fcluster_criterion`.
    fcluster_criterion : str
        The criterion to use in forming flat clusters. Supported criteria include `"inconsistent"`, 
        `"distance"`, and `"maxclust"`.
    fcluster_depth : int
        The maximum depth to perform inconsistency calculation if `fcluster_criterion="inconsistent"`. 
        Ignored for other criteria.

    Methods
    -------
    fit(source_data: np.ndarray) -> np.ndarray
        Fits the hierarchical clustering model to the provided data and computes cluster labels.

    Parameters
    ----------
    linkage_method : str, optional
        Linkage method for clustering. Default is `"complete"`.
    linkage_metric : str, optional
        Distance metric for clustering. Default is `"cosine"`.
    linkage_optimal_ordering : bool, optional
        Whether to reorder linkage matrix for optimal leaf ordering. Default is `False`.
    fcluster_t : float, optional
        Threshold for forming flat clusters. Default is `0.25`.
    fcluster_criterion : str, optional
        Criterion for forming flat clusters. Default is `"distance"`.
    fcluster_depth : int, optional
        Depth for inconsistency calculation when `fcluster_criterion="inconsistent"`. Default is `2`.

    Examples
    --------
    >>> import numpy as np
    >>> source_data = np.random.rand(10, 5)
    >>> model = SCH()
    >>> labels = model.fit(source_data)
    >>> print(labels)
    [1 1 2 2 3 3 4 4 5 5]
    '''

    def __init__(
            self,
            linkage_method: str = "complete",
            linkage_metric: str = "cosine",
            linkage_optimal_ordering: bool = False,
            fcluster_t: float = 0.25,
            fcluster_criterion: str = "distance",
            fcluster_depth: int = 2,
        ):

        self.labels = None

        self.linkage_method = linkage_method
        self.linkage_metric = linkage_metric
        self.linkage_optimal_ordering = linkage_optimal_ordering

        self.fcluster_t = fcluster_t
        self.fcluster_criterion = fcluster_criterion
        self.fcluster_depth = fcluster_depth


    def fit(self, source_data: np.ndarray):
        if source_data.shape[0] > 32000:
            raise ValueError(f'Very large sample! Recommended no more than 32000 samples. Submitted: {source_data.shape[0]}.')

        Z = sch.linkage(
            y=source_data,
            method=self.linkage_method,
            metric=self.linkage_metric,
            optimal_ordering=self.linkage_optimal_ordering,
        )

        self.labels = sch.fcluster(
            Z=Z,
            t=self.fcluster_t,
            criterion=self.fcluster_criterion,
            depth=self.fcluster_depth,
        )

        return self.labels


class HDBSCAN():
    '''
    A wrapper class for the HDBSCAN clustering algorithm.

    The `HDBSCAN` class simplifies the usage of the HDBSCAN clustering algorithm by encapsulating 
    the core functionality of the `hdbscan.HDBSCAN` class. It provides a streamlined interface for 
    clustering data and retrieving cluster labels.

    Attributes
    ----------
    labels : np.ndarray or None
        Cluster labels assigned to each data point after fitting the model. Initially set to `None`.

    Methods
    -------
    fit(source_data: np.ndarray) -> np.ndarray
        Fits the HDBSCAN clustering model to the provided data and computes cluster labels.

    Parameters
    ----------
    min_cluster_size : int, optional
        The minimum size of clusters. Default is 5.
    min_samples : int, optional
        The minimum number of samples in a neighborhood for a point to be considered a core point. 
    cluster_selection_epsilon : float, optional
        The distance threshold for cluster selection. Default is 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> from hdbscan import HDBSCAN
    >>> source_data = np.random.rand(100, 5)
    >>> model = HDBSCAN(min_cluster_size=10)
    >>> labels = model.fit(source_data)
    >>> print(labels)
    [0 0 1 -1 2 2 1 1 3 ...]
    '''
    def __init__(self, **kwargs):

        self.__hdbscan__ = source_HDBSCAN(**kwargs)
        self.labels = None

    def fit(self, source_data: np.ndarray):
        if source_data.shape[0] > 64000:
            raise ValueError(f'Very large sample! Recommended no more than 64000 samples. Submitted: {source_data.shape[0]}.')

        self.labels = self.__hdbscan__.fit_predict(source_data)

        return self.labels