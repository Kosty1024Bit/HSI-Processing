import numpy as np

from scipy.spatial.distance import cdist


def get_centroids_and_medoids(labels, data, metric='cosine'):
    '''
    Вычисление центроидов и медианов для кластеров в размеченных данных.

    Функция вычисляет центроиды и медианы для каждого кластера в предоставленных данных, 
    учитывая метки кластеров. Центроиды — это среднее значение всех точек в кластере, 
    а медианы определяются как точка кластера, наиболее близкая к центроиду, 
    в соответствии с заданной метрикой расстояния.

    Параметры
    ---------
    labels : np.ndarray
        1D массив меток кластеров для точек данных. Точки с одинаковой меткой 
        считаются принадлежащими одному и тому же кластеру.
    data : np.ndarray
        2D массив формы `(n_samples, n_features)`, содержащий точки данных.
    metric : str, optional
        Метрика расстояния, используемая для вычисления медианов. По умолчанию `'cosine'`.
        Другие допустимые метрики включают `'euclidean'`, `'manhattan'` и другие, поддерживаемые 
        функцией `scipy.spatial.distance.cdist`.

    Возвращаемые значения
    ---------------------
    centroids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий центроиды кластеров.
    medoids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий медианы кластеров.

    Примеры
    -------
    Вычисление центроидов и медианов для косинусного сходства

    >>> import numpy as np
    >>> from hsip.analysis.analysis import get_centroids_and_medoids
    >>> labels = np.array([0, 0, 1, 1, 2, 2])
    >>> data = np.array([
    ...     [1, 2], [2, 3], 
    ...     [3, 4], [4, 5], 
    ...     [5, 6], [6, 7]
    ... ])
    >>> centroids, medoids = get_centroids_and_medoids(labels, data, metric='cosine')
    >>> print("Центроиды:")
    >>> print(centroids)
    Центроиды:
    [[1.5 2.5]
     [3.5 4.5]
     [5.5 6.5]]
    >>> print("Медианы:")
    >>> print(medoids)
    Медианы:
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
    Вычисление матрицы перекрёстной корреляции для заданного набора данных.

    Эта функция вычисляет попарные расстояния между всеми строками входных данных 
    с использованием указанной метрики расстояния и возвращает полученную матрицу 
    перекрёстной корреляции.

    Параметры
    ---------
    data : np.ndarray
        2D массив формы `(n_samples, n_features)`, содержащий входные данные, где `n_samples` — 
        количество точек данных, а `n_features` — количество признаков.
    metric : str, optional
        Метрика расстояния, используемая для вычисления попарных расстояний. По умолчанию `'euclidean'`.
        Поддерживаемые метрики включают `'euclidean'`, `'manhattan'`, `'cosine'` и другие, 
        доступные в `scipy.spatial.distance.cdist`.

    Возвращаемые значения
    ---------------------
    cross_corr_mat : np.ndarray
        2D массив формы `(n_samples, n_samples)`, содержащий попарные расстояния между всеми 
        точками данных во входных данных.

    Примеры
    -------
    Вычисление матрицы перекрёстной корреляции с метрикой Евклида по умолчанию
    
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