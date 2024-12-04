import numpy as np

from scipy.spatial.distance import cdist


def get_centroids_and_medoids(labels, data, metric='cosine'):
    '''
    Compute centroids and medoids for clusters in labeled data.

    This function calculates the centroids and medoids for each cluster in the provided data, 
    given cluster labels. Centroids are the mean of all points in a cluster, while medoids
    are calculated as the point in the cluster closest to the centroid based on the
    specified distance metric.
    
    Parameters
    ----------
    labels : np.ndarray
        1D array of cluster labels for the data points. Points with the same label are considered 
        to belong to the same cluster.
    data : np.ndarray
        2D array of shape `(n_samples, n_features)` containing the data points.
    metric : str, optional
        The distance metric used to calculate the medoids. Default is `'cosine'`. 
        Other valid metrics include `'euclidean'`, `'manhattan'`, and others supported by 
        `scipy.spatial.distance.cdist`.

    Returns
    -------
    centroids : np.ndarray
        2D array of shape `(n_clusters, n_features)` containing the centroids of the clusters.
    medoids : np.ndarray
        2D array of shape `(n_clusters, n_features)` containing the medoids of the clusters.

    Examples
    --------
    Compute centroids and medoids for cosine similarity
    >>> import numpy as np
    >>> from hsip.analysis.analysis import get_centroids_and_medoids
    >>> labels = np.array([0, 0, 1, 1, 2, 2])
    >>> data = np.array([
    ...     [1, 2], [2, 3], 
    ...     [3, 4], [4, 5], 
    ...     [5, 6], [6, 7]
    ... ])
    >>> centroids, medoids = get_centroids_and_medoids(labels, data, metric='cosine')
    >>> print("Centroids:")
    >>> print(centroids)
    Centroids:
    [[1.5 2.5]
     [3.5 4.5]
     [5.5 6.5]]
    >>> print("Medoids:")
    >>> print(medoids)
    Medoids:
    [[1. 2.]
     [3. 4.]
     [5. 6.]]
    '''
    unique_labels = np.unique(labels)
    centroids = np.zeros(shape=(unique_labels.shape[0], data.shape[-1]), dtype=float)
    medoids = np.zeros(shape=(unique_labels.shape[0], data.shape[-1]), dtype=float)
    
    for i, lbl in enumerate(unique_labels):
        cluster = data[labels == lbl]
        centroids[i] = cluster.mean(axis=0)
        
        dist = cdist(cluster, centroids[i, np.newaxis], metric=metric)[:, 0]
        medoids[i] = cluster[dist.argmin()]
        
    return centroids, medoids


def get_cross_correlation_matrix(data: np.ndarray, metric: str='euclidean'):
    '''
    Compute the cross-correlation matrix for a given dataset.

    This function calculates the pairwise distances between all rows in the input data 
    using the specified distance metric and returns the resulting cross-correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape `(n_samples, n_features)` containing the input data, where `n_samples` 
        is the number of data points and `n_features` is the number of features.
    metric : str, optional
        The distance metric to use for calculating pairwise distances. Default is `'euclidean'`. 
        Supported metrics include `'euclidean'`, `'manhattan'`, `'cosine'`, and others supported by 
        `scipy.spatial.distance.cdist`.

    Returns
    -------
    cross_corr_mat : np.ndarray
        2D array of shape `(n_samples, n_samples)` containing the pairwise distances between 
        all data points in the input data.

    Examples
    --------
    Compute cross-correlation matrix with the default Euclidean metric
    >>> import numpy as np
    >>> from hsip.analysis.analysis import get_cross_correlation_matrix
    >>> data = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    >>> cross_corr_mat = get_cross_correlation_matrix(data)
    >>> print(cross_corr_mat)
    [[0.         1.41421356 2.82842712 4.24264069]
     [1.41421356 0.         1.41421356 2.82842712]
     [2.82842712 1.41421356 0.         1.41421356]
     [4.24264069 2.82842712 1.41421356 0.        ]]
    '''
    
    if data.shape[0] > 16:
        raise ValueError('The matrix will be too large to analyze.')
        
    cross_corr_mat = cdist(data, data, metric=metric)
    
    return cross_corr_mat