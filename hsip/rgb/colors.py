import numpy as np


class Color:
    '''
    A utility class for managing and accessing a predefined set of colors.

    This class provides a collection of RGB color values in the range [0, 1], 
    and methods to retrieve single or multiple colors by index or slice. 
    It also supports modular indexing to wrap around the color set.

    Attributes
    ----------
    colors : np.ndarray
        A 2D array of shape `(n_colors, 3)` containing RGB color values in the range [0, 1].

    Methods
    -------
    __get_color__(index)
        Retrieves a single color or a subset of colors by index or slice.
    __getitem__(index)
        Alias for `__get_color__`, allows indexing with square brackets.
    __len__()
        Returns the total number of colors in the palette.
    __iter__()
        Returns an iterator over the colors.

    Parameters
    ----------
    None

    Examples
    --------
    Create an instance of the `Color` class and retrieve colors:
    
    >>> from hsip.rgb.colors import Color
    >>> color_palette = Color()

    Access a single color using an index:
    >>> color_palette[0]
    array([0.90196078, 0.09803922, 0.29411765])

    Access multiple colors using a slice:
    >>> color_palette[1:4]
    array([[0.23529412, 0.70588235, 0.29411765],
           [0.        , 0.50980392, 0.78431373],
           [0.96078431, 0.50980392, 0.18823529]])

    Modular indexing wraps around:
    >>> color_palette[20]
    array([0.23529412, 0.70588235, 0.29411765])  # Index 20 maps to 1 (20 % len(colors))

    Iterate through the colors:
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