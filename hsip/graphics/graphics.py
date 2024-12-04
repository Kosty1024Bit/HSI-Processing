import numpy as np

import matplotlib.pyplot as plt

def show_correlation_matrix(corr_matrix: np.ndarray, label_axis: list, path: str = None):
    '''
    Отображение матрицы корреляции с аннотированными значениями.

    Эта функция визуализирует матрицу корреляции с аннотированными значениями, где каждый элемент матрицы
    отображается на соответствующей позиции с точностью до двух знаков после запятой.

    Параметры
    ---------
    corr_matrix : np.ndarray
        Квадратная матрица корреляции, которая будет отображена в виде тепловой карты.
    label_axis : list
        Список меток для осей X и Y, который используется для подписей в графике.
    path : str, optional
        Путь, по которому сохранить изображение. Если параметр не указан (по умолчанию `None`), то изображение 
        будет показано в окне.

    Возвращает
    ----------
    None

    Пример
    ------
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