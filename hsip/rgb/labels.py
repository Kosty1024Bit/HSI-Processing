import numpy as np
from hsip.rgb.colors import colors_set

def labels_to_rgb(labels: np.ndarray, RGB_image: np.ndarray = None):
    '''
    Преобразует метки в маске в представление в виде изображения RGB.

    Эта функция генерирует RGB-изображение, где каждой уникальной метке из входного массива `labels` 
    присваивается определенный цвет. Если предоставлено `RGB_image`, цвет каждой метки вычисляется как 
    среднее значение цветов для соответствующих областей в `RGB_image`. Если изображение не предоставлено, 
    используются заранее определенные цвета из глобального набора `colors_set`. Внимание: в наборе `colors_set` 
    только 17 цветов, и если уникальных меток больше, то цвета для некоторых из них будут повторяться!

    Параметры
    ---------
    labels : np.ndarray
        2D или 3D массив, представляющий маску меток. Каждое уникальное значение соответствует различному классу.
    RGB_image : np.ndarray, необязательный
        3D массив формы `(height, width, 3)`, представляющий существующее RGB-изображение. Если предоставлено, 
        цвет для каждой метки вычисляется как среднее значение RGB для соответствующих регионов в этом изображении.

    Возвращает
    ----------
    np.ndarray
        RGB-изображение формы `(height, width, 3)` с присвоенными цветами для каждой метки.

    Примеры
    -------
    Пример 1: Использование заранее определенного набора цветов:

    >>> from hsip.rgb.labels import labels_to_rgb
    >>> labels = np.array([[0, 0, 1],
    ...                    [1, 2, 2]])
    >>> rgb_image = labels_to_rgb(labels)
    >>> print(rgb_image.shape)
    (2, 3, 3)

    Пример 2: Использование существующего RGB-изображения для вычисления цветов:
    
    >>> from hsip.rgb.labels import labels_to_rgb
    >>> labels = np.array([[0, 0, 1],
    ...                    [1, 2, 2]])
    >>> RGB_image = np.random.randint(0, 255, size=(2, 3, 3), dtype=np.uint8)
    >>> rgb_image = labels_to_rgb(labels, RGB_image=RGB_image)
    >>> print(rgb_image.shape)
    (2, 3, 3)
    '''
    
    if RGB_image is not None:
        if labels.shape != RGB_image.shape[:-1]:
            raise ValueError('The sizes of `labels` and `RGB_image` must match.')

    RGB_syntes = np.zeros(shape=(list(labels.shape) + [3]), dtype=np.uint8)
    unique_labels = np.unique(labels)
    
    for i, lbl in enumerate(unique_labels):
        class_mask = labels == lbl

        if RGB_image is not None:
            color_class = [np.mean(RGB_image[class_mask, b]) for b in range(3)]
        else:
            color_class = np.uint8(colors_set[i] * 255)
        
        RGB_syntes[class_mask] = color_class

    return RGB_syntes