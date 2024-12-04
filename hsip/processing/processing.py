import numpy as np
from tqdm import tqdm

def normalize(array: np.ndarray):
    '''
    Оболочка для алгоритма кластеризации HDBSCAN.

    Класс `HDBSCAN` упрощает использование алгоритма HDBSCAN, инкапсулируя основную функциональность 
    класса `hdbscan.HDBSCAN`. Он предоставляет удобный интерфейс для кластеризации данных и получения меток кластеров.

    Атрибуты
    --------
    labels : np.ndarray or None
        Метки кластеров, присвоенные каждой точке данных после обучения модели. Изначально установлено в `None`.
    centroids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий центроиды кластеров.
    medoids : np.ndarray
        2D массив формы `(n_clusters, n_features)`, содержащий медианы кластеров.

    Методы
    ------
    fit(source_data: np.ndarray) -> np.ndarray
        Обучает модель HDBSCAN на предоставленных данных и вычисляет метки кластеров.

    Параметры
    ---------
    min_cluster_size : int, optional
        Минимальный размер кластеров. По умолчанию 5.
    min_samples : int, optional
        Минимальное количество точек в окрестности, чтобы точка считалась ядровой.
    cluster_selection_epsilon : float, optional
        Порог расстояния для выбора кластеров. По умолчанию 0.0.

    Примеры
    -------
    >>> import numpy as np
    >>> from hsip.clustering.clustering import HDBSCAN
    >>> source_data = np.random.rand(100, 5)
    >>> model = HDBSCAN(min_cluster_size=10)
    >>> labels = model.fit(source_data)
    >>> print(labels)
    [0 0 1 -1 2 2 1 1 3 ...]
    '''
    
    return (array - array.min()) / (array.max() - array.min())


def rayleigh_scattering(spectral_data: np.ndarray, inplace=False, verbose=True):
    '''
    Вычисление сигнатуры рассеяния Рэлея для спектрального набора данных.

    Параметры
    ---------
    spectral_data : np.ndarray
        Многомерный массив NumPy, где последняя размерность представляет спектральные диапазоны.
    inplace : bool, default=False
        Если `True`, функция изменяет `spectral_data`, вычитая вычисленную сигнатуру Рэлея.
    verbose : bool, verbose=True
        Если `True`, отображает индикатор прогресса.

    Возвращаемые значения
    ---------------------
    np.ndarray
        1D массив, содержащий сигнатуру рассеяния Рэлея для каждого спектрального диапазона.

    Примеры
    -------
    Применение фильтра Рэлея к 3D спектральному набору данных:

    >>> import numpy as np
    >>> from hsip.processing.processing import rayleigh_scattering
    >>> data = np.random.rand(100, 100, 10) * 10  # Пример спектральных данных
    >>> rayleigh_signature = rayleigh_scattering(data, inplace=True)
    >>> rayleigh_signature  # Сигнатура рассеяния Рэлея
    array([...])
    '''
    
    if not isinstance(spectral_data, np.ndarray):
        raise TypeError('spectral_data must be an np.ndarray')
        
    bands = spectral_data.shape[-1]
    rayleigh_offsets = np.zeros(shape=bands, dtype=spectral_data.dtype)
    for i in tqdm(range(bands), disable=not verbose):
        layer = spectral_data[..., i]
        rayleigh_offsets[i] = layer[layer > 0].min()
    
    if inplace:
        spectral_data -= rayleigh_offsets
        spectral_data[spectral_data < 0] = 0
        
    return rayleigh_offsets


def sigma_maximum_filter(spectral_data: np.ndarray, sigma: float = 3, thresholds: np.ndarray = None):
    '''
    Применяет сигма-ориентированный максимальный фильтр к входным спектральным данным, ограничивая значения 
    на основе вычисленного порога: среднее значение + сигма * стандартное отклонение.

    Параметры
    ---------
    spectral_data : np.ndarray
        Входной массив спектральных данных. Ожидается, что последняя размерность соответствует спектральным диапазонам.
    sigma : float, optional
        Множитель для стандартного отклонения, используемый при расчете порога. По умолчанию 3.
    thresholds : np.ndarray, optional
        Массив для хранения вычисленных порогов. Если передан, его форма должна совпадать с последней 
        размерностью `spectral_data`. Если `None`, пороги вычисляются и возвращаются внутренне. По умолчанию None.

    Возвращаемые значения
    ---------------------
    np.ndarray
        Отфильтрованные спектральные данные с ограниченными значениями на основе вычисленных порогов.

    Примеры
    -------
    Применение сигма-фильтра к 3D спектральному набору данных:

    >>> import numpy as np
    >>> from hsip.processing.processing import sigma_maximum_filter
    >>> data = np.random.rand(100, 100, 10) * 10  # Пример спектральных данных
    >>> thresholds = np.zeros(data.shape[-1:], dtype=np.float32)
    >>> result_with_thresholds = sigma_maximum_filter(data, sigma=2, thresholds=thresholds)
    >>> thresholds  # Обновленные пороги
    array([...])
    '''
    
    if not isinstance(spectral_data, np.ndarray):
        raise TypeError('spectral_data must be an np.ndarray')
    
    axes = tuple(range(len(spectral_data.shape) - 1))
    layer_mean = spectral_data.mean(axis=axes, keepdims=True)
    layer_std = spectral_data.std(axis=axes, keepdims=True)
    
    if thresholds is None:
        thresholds = np.zeros(shape=spectral_data.shape[-1:])
    elif thresholds.shape != spectral_data.shape[-1:]:
        raise ValueError("Provided thresholds must have the same shape as the spectral axis of the data.")
    thresholds[...] = layer_mean + (layer_std * sigma)
    
    return np.minimum(spectral_data, thresholds)