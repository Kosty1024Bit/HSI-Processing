import numpy as np

from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch

from hdbscan import HDBSCAN as source_HDBSCAN

from tqdm import tqdm

from hsip.analysis.analysis import get_centroids_and_medoids


class CosClust():
    '''
    Кластеризационный алгоритм на основе косинусного сходства. Объединяет образцы в кластеры, основываясь на 
    пороговом значении косинусного сходства, и назначает метки каждому образцу.

    Параметры
    ---------
    threshold : float, optional
        Порог косинусного сходства для определения принадлежности к кластеру. По умолчанию 0.9.
    verbose : bool, optional
        Если установлено в `True`, отображает процесс и дополнительную информацию во время кластеризации. 
        По умолчанию `True`.

    Атрибуты
    --------
    threshold : float
        Порог косинусного сходства для кластеризации.
    labels : np.ndarray or None
        Метки кластеров, назначенные каждому образцу. Инициализируется как `None`, заполняется после вызова метода `fit`.
    reference_set : list
        Список эталонных образцов, представляющих каждый кластер.
    centroids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий центроиды кластеров.
    medoids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий медианы кластеров.

    Методы
    ------
    fit(source_data)
        Выполняет кластеризацию входных данных и возвращает метки кластеров.

    Примеры
    -------
    >>> import numpy as np
    >>> from hsip.clustering.clustering import CosClust
    >>> data = np.random.rand(100, 50)  # 100 образцов, каждый с 50 признаками
    >>> model = CosClust(threshold=0.8, verbose=True)
    >>> labels = model.fit(data)
    >>> print(labels)
    array([0, 1, 0, 2, ..., 1])
    '''
        
    def __init__(self, threshold: float = 0.9, verbose=True):
        self.threshold = threshold
        self.labels = None
        self.verbose = verbose
        self.reference_set = []
        
        self.centroids = None
        self.medoids = None
        

    def fit(self, source_data: np.ndarray):

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
        
        self.centroids, self.medoids = get_centroids_and_medoids(self.labels, source_data, 'cosine')

        return self.labels


class SCH():
    '''
    Класс для выполнения иерархической кластеризации с использованием методов `linkage` и `fcluster` из библиотеки SciPy.

    Класс `SCH` предоставляет интерфейс для иерархической кластеризации с настраиваемыми параметрами объединения и кластеризации.

    Атрибуты
    --------
    labels : np.ndarray or None
        Метки кластеров, присвоенные каждой точке данных после обучения модели. Изначально установлено в `None`.
    linkage_method : str
        Метод объединения, используемый для иерархической кластеризации. Поддерживаемые методы включают 
        `"single"`, `"complete"`, `"average"`, `"weighted"`, `"centroid"`, `"median"`, и `"ward"`.
    linkage_metric : str
        Метрика расстояния, используемая для вычисления попарных расстояний между точками данных. 
        Популярные метрики: `"euclidean"`, `"cosine"`, `"cityblock"`, `"hamming"`.
    linkage_optimal_ordering : bool
        Если `True`, матрица объединений будет упорядочена для минимизации расстояний между последовательными листьями.
    fcluster_t : float
        Порог для формирования плоских кластеров. Значение `t` зависит от `fcluster_criterion`.
    fcluster_criterion : str
        Критерий для формирования плоских кластеров. Поддерживаемые критерии: `"inconsistent"`, 
        `"distance"`, `"maxclust"`.
    fcluster_depth : int
        Максимальная глубина для расчета несогласованности, если `fcluster_criterion="inconsistent"`. 
        Игнорируется для других критериев.
    centroids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий центроиды кластеров.
    medoids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий медианы кластеров.

    Методы
    ------
    fit(source_data: np.ndarray) -> np.ndarray
        Обучает модель иерархической кластеризации на предоставленных данных и вычисляет метки кластеров.

    Параметры
    ---------
    linkage_method : str, optional
        Метод объединения для кластеризации. По умолчанию `"complete"`.
    linkage_metric : str, optional
        Метрика расстояния для кластеризации. По умолчанию `"cosine"`.
    linkage_optimal_ordering : bool, optional
        Перестраивать ли матрицу объединений для оптимального порядка листьев. По умолчанию `False`.
    fcluster_t : float, optional
        Порог для формирования плоских кластеров. По умолчанию `0.25`.
    fcluster_criterion : str, optional
        Критерий для формирования плоских кластеров. По умолчанию `"distance"`.
    fcluster_depth : int, optional
        Глубина для расчета несогласованности, если `fcluster_criterion="inconsistent"`. По умолчанию `2`.

    Примеры
    -------
    >>> import numpy as np
    >>> from hsip.clustering.clustering import SCH
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
        
        self.centroids = None
        self.medoids = None


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

        self.centroids, self.medoids = get_centroids_and_medoids(self.labels, source_data, self.linkage_metric)
        
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
    centroids : np.ndarray
        2D array of shape `(n_clusters, n_features)` containing the centroids of the clusters.
    medoids : np.ndarray
        2D array of shape `(n_clusters, n_features)` containing the medoids of the clusters.

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
    >>> from hsip.clustering.clustering import HDBSCAN
    >>> source_data = np.random.rand(100, 5)
    >>> model = HDBSCAN(min_cluster_size=10)
    >>> labels = model.fit(source_data)
    >>> print(labels)
    [0 0 1 -1 2 2 1 1 3 ...]
    '''
    def __init__(self, **kwargs):
        self.__hdbscan__ = source_HDBSCAN(**kwargs)
        self.labels = None
        
        self.centroids = None
        self.medoids = None

        
    def fit(self, source_data: np.ndarray):
        if source_data.shape[0] > 64000:
            raise ValueError(f'Very large sample! Recommended no more than 64000 samples. Submitted: {source_data.shape[0]}.')

        self.labels = self.__hdbscan__.fit_predict(source_data)
        
        self.centroids, self.medoids = get_centroids_and_medoids(self.labels, source_data, 'euclidean')

        return self.labels