import numpy as np


class Color:
    '''
    Утилитный класс для управления и доступа к предустановленному набору цветов.

    Этот класс предоставляет коллекцию значений цветов в формате RGB в диапазоне [0, 1] 
    и методы для получения одного или нескольких цветов по индексу или срезу. 
    Он также поддерживает модульную индексацию для циклического обращения к набору цветов.

    Атрибуты
    --------
    colors : np.ndarray
        Двумерный массив формы `(n_colors, 3)`, содержащий значения цветов RGB в диапазоне [0, 1].

    Методы
    ------
    __get_color__(index)
        Получает один цвет или подмножество цветов по индексу или срезу.
    __getitem__(index)
        Псевдоним для метода `__get_color__`, позволяет использовать квадратные скобки для индексации.
    __len__()
        Возвращает общее количество цветов в палитре.
    __iter__()
        Возвращает итератор по цветам.

    Параметры
    ----------
    Нет

    Примеры
    -------
    Создание экземпляра класса `Color` и получение цветов:

    >>> from hsip.rgb.colors import Color
    >>> color_palette = Color()

    Получение одного цвета с использованием индекса:

    >>> color_palette[0]
    array([0.90196078, 0.09803922, 0.29411765])

    Получение нескольких цветов с использованием среза:
    
    >>> color_palette[1:4]
    array([[0.23529412, 0.70588235, 0.29411765],
           [0.        , 0.50980392, 0.78431373],
           [0.96078431, 0.50980392, 0.18823529]])

    Модульная индексация:
    
    >>> color_palette[20]
    array([0.23529412, 0.70588235, 0.29411765])  # Индекс 20 соответствует индексу 1 (20 % len(colors))

    Итерация по цветам:
    
    >>> for color in color_palette:
    ...     print(color)
    array([0.90196078, 0.09803922, 0.29411765])
    array([0.23529412, 0.70588235, 0.29411765])
    ...
    '''

    def __init__(self):
        self.colors = np.array([
            [0.90196078, 0.09803922, 0.29411765],
            [0.23529412, 0.70588235, 0.29411765],
            [0.        , 0.50980392, 0.78431373],
            [0.96078431, 0.50980392, 0.18823529],
            [0.56862745, 0.11764706, 0.70588235],
            [0.2745098 , 0.94117647, 0.94117647],
            [0.94117647, 0.19607843, 0.90196078],
            [0.82352941, 0.96078431, 0.23529412],
            [0.98039216, 0.74509804, 0.83137255],
            [0.        , 0.50196078, 0.50196078],
            [0.66666667, 0.43137255, 0.15686275],
            [0.50196078, 0.        , 0.        ],
            [0.66666667, 1.        , 0.76470588],
            [0.50196078, 0.50196078, 0.        ],
            [1.        , 0.84313725, 0.70588235],
            [0.        , 0.        , 0.50196078],
            [0.50196078, 0.50196078, 0.50196078],
        ], dtype=float)


    def __get_color__(self, index: int | slice) -> int | slice:
        len_array = len(self.colors) # Number of colors

        attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
        if all(hasattr(index, attr) for attr in attrs):
            index = int(index)

        if type(index) is int:
            return self.colors[index % len_array]

        if type(index) is slice:
            if index.start is not None and index.stop is None:
                raise ValueError("Invalid slice values, if start is specified, stop must be specified.")

            return np.array([self.colors[i % len_array] for i in range(index.start, index.stop)], dtype=float)


    def __getitem__(self, index):
        return self.__get_color__(index)


    def __len__(self):
        return len(self.colors)


    def __iter__(self):
        return iter(np.array(self.colors, dtype=float))

colors_set = Color()