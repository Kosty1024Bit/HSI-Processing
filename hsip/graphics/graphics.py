import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

from hsip.rgb.colors import Color


def show_correlation_matrix(corr_matrix: np.ndarray, label_axis: list, path: str = None):
    '''
    Отображение матрицы корреляции с аннотированными значениями.

    Эта функция визуализирует матрицу корреляции с аннотированными значениями, где каждый элемент матрицы
    отображается на соответствующей позиции с точностью до двух знаков после запятой.

    Параметры
    ----------
    corr_matrix : np.ndarray
        Квадратная матрица корреляции, которая будет отображена в виде тепловой карты.
    label_axis : list
        Список меток для осей X и Y, который используется для подписей в графике.
    path : str, optional
        Путь, по которому сохранить изображение. Если параметр не указан (по умолчанию `None`), то изображение 
        будет показано в окне.

    Возвращает
    ---------
    None

    Пример
    -------
    >>> import numpy as np
    >>> corr_matrix = np.array([[1, 0.8], [0.8, 1]])
    >>> label_axis = ['A', 'B']
    >>> show_correlation_matrix(corr_matrix, label_axis)
    '''
    
    fig, ax = plt.subplots(figsize = (9, 9))
    im = ax.imshow(corr_matrix)

    ax.set_xticks(np.arange(len(label_axis)), labels=label_axis, fontsize = 12)
    ax.set_yticks(np.arange(len(label_axis)), labels=label_axis, fontsize = 12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    for i in range(len(label_axis)):
        for j in range(len(label_axis)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center", color="w", fontsize = 16)
    
    fig.tight_layout()
    
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()


def show_image(image: np.ndarray, path: str = None, color_bar: bool = False, scale_img: float = 1.0):
    '''
    Отображение изображения с возможностью сохранения и масштабирования.

    Функция визуализирует переданное изображение. Поддерживается отображение
    цветных RGB изображений и одноканальных изображений с настройкой цветовой шкалы.
    Также можно сохранить изображение в файл, указав путь в параметре `path`.

    Параметры
    ----------
    image : np.ndarray
        Массив изображения. Для RGB изображения тип данных должен быть `uint8`, 
        а размерность массива — `(height, width, 3)`. Для одноканального изображения
        допускаются другие типы данных и размерность `(height, width)`.
    path : str, optional
        Путь для сохранения изображения. Если не указан (по умолчанию `None`), 
        изображение будет отображено в окне.
    color_bar : bool, optional
        Если `True`, для одноканальных изображений добавляется цветовая шкала (по умолчанию `False`).
        Для RGB изображений этот параметр игнорируется.
    scale_img : float, optional
        Коэффициент масштабирования изображения. Например, при значении 2.0 изображение
        будет отображено в два раза крупнее (по умолчанию `1.0`).

    Возвращает
    ---------
    None

    Пример
    -------
    Отображение и сохранение RGB изображения:
    
    >>> import numpy as np
    >>> rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> show_image(rgb_image, path='rgb_image.png', scale_img=1.5)

    Отображение одноканального изображения с цветовой шкалой:
    
    >>> grayscale_image = np.random.rand(100, 100)
    >>> show_image(grayscale_image, color_bar=True)
    '''
    
    dpi = rcParams['figure.dpi']
    height = image.shape[0]
    width = image.shape[1]
    fig_size = (width / float(dpi)) * scale_img, (height / float(dpi)) * scale_img
    
    if len(image.shape) == 3:
        color_bar = False
        if image.dtype != np.uint8:
            raise ValueError('For color RGB image the type should be uint8.')
        
    plt.figure(figsize=fig_size)
    plt.imshow(image)
    if color_bar:
        plt.colorbar(shrink=0.75)
    plt.axis('off')
    
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()


def show_curves(
    curves: np.ndarray,
    labels: list | np.ndarray = None,
    colors: list | np.ndarray = None,
    xlabel: str = 'Номер канала',
    ylabel: str = 'Величина',
    title: str = None,
    path: str = None, 
    scale_img: float = 4.0
):
    '''
    Отображение набора кривых с возможностью настройки меток, цветов и сохранения.

    Функция визуализирует набор кривых, где каждая строка массива `curves` соответствует одной кривой.
    Можно задавать метки для легенды, цвета кривых и параметры отображения.

    Параметры
    ----------
    curves : np.ndarray
        Двумерный массив размером `(n_curves, n_points)`, где каждая строка представляет одну кривую.
        Если передан одномерный массив, он автоматически преобразуется в массив с одной строкой.
    labels : list | np.ndarray, optional
        Метки для каждой кривой. Если не указаны (по умолчанию `None`), кривые будут отображены без легенды.
    colors : list | np.ndarray, optional
        Цвета для кривых. Если не указаны (по умолчанию `None`), используются заранее заданные цвета.
        Количество цветов должно совпадать с количеством кривых.
    xlabel : str, optional
        Метка оси X (по умолчанию `'Номер канала'`).
    ylabel : str, optional
        Метка оси Y (по умолчанию `'Величина'`).
    title : str, optional
        Заголовок графика (по умолчанию `None`).
    path : str, optional
        Путь для сохранения изображения. Если не указан (по умолчанию `None`), изображение отображается в окне.
    scale_img : float, optional
        Коэффициент масштабирования размера изображения (по умолчанию `4.0`).

    Возвращает
    ---------
    None

    Пример
    -------
    Отображение трёх кривых с разными метками и цветами:
    
    >>> import numpy as np
    >>> curves = np.random.rand(3, 100)
    >>> labels = ['Кривая 1', 'Кривая 2', 'Кривая 3']
    >>> colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> show_curves(curves, labels=labels, colors=colors, title='Пример графика')

    Сохранение графика в файл:
    
    >>> show_curves(curves, path='curves_plot.png')
    '''
    
    if len(curves.shape) == 1:
        curves = curves[np.newaxis]
    
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels, list):
            labels = np.array(labels)
    
    dpi = rcParams['figure.dpi']
    fig_size = (160 / float(dpi)) * scale_img, (90 / float(dpi)) * scale_img
    
    if colors is None:
        colors = Color() # Заранее определённый набор цветов
    else:
        if isinstance(colors, list):
            colors = np.array(colors)
        if len(colors.shape) == 1:
            colors = colors[np.newaxis]
        if curves.shape[0] != colors.shape[0]:
            print(colors)
            raise ValueError('The number of curves and colors for them must match.')
    
    plt.figure(figsize=fig_size)
    for i in range(curves.shape[0]):
    
        if labels is None:
            plt.plot(curves[i], color=colors[i], lw=2)
        else:
            plt.plot(curves[i], color=colors[i], lw=2, label=labels[i])
        
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if labels is not None:
        plt.legend()
    
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()