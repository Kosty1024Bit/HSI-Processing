import numpy as np

from hsip.processing.processing import normalize


def hsi_synthesize_rgb(spectral_data: np.ndarray, rgb_bands: list | np.ndarray = None, wavelengths: list | np.ndarray = None):
    """
    Синтезирует RGB-изображение из гиперспектральных данных.

    Параметры
    ----------
    spectral_data : np.ndarray
        Гиперспектральные данные изображения, ожидается, что это 3D массив формы (height, width, bands).
    rgb_bands : list of int, необязательный
        Список, содержащий индексы полос для каналов красного, зеленого и синего цветов, в указанном порядке.
        Должен иметь длину 3. Если указан, `wavelengths` игнорируется.
    wavelengths : list of float, необязательный
        Список, содержащий длины волн, соответствующие каждой полосе в `spectral_data`.
        Если указан, выбираются наиболее близкие длины волн к 650 нм (красный), 550 нм (зеленый) и 450 нм (синий).

    Возвращает
    ---------
    np.ndarray
        3D массив формы (height, width, 3), представляющий синтезированное RGB-изображение.

    Примеры
    --------
    Использование индексов полос напрямую:

    >>> from hsip.rgb.rgb import hsi_synthesize_rgb
    >>> spectral_data = np.random.rand(100, 100, 224)  # Пример гиперспектральных данных
    >>> rgb_bands = [50, 100, 150]  # Пример полос для красного, зеленого, синего
    >>> rgb_image = synthesize_rgb(spectral_data, rgb_bands=rgb_bands)
    >>> print(rgb_image.shape)
    (100, 100, 3)

    Использование длин волн:

    >>> wavelengths = np.linspace(400, 700, 224)  # Пример данных о длинах волн
    >>> rgb_image = synthesize_rgb(spectral_data, wavelengths=wavelengths)
    >>> print(rgb_image.shape)
    (100, 100, 3)
    """
    if rgb_bands is None and wavelengths is None:
        raise ValueError("Either `rgb_bands` or `wavelengths` must be provided.")

    if rgb_bands is not None:
        if isinstance(rgb_bands, np.ndarray):
            rgb_bands = rgb_bands.tolist()

        if len(rgb_bands) != 3:
            raise ValueError("`rgb_bands` must contain exactly three indices for red, green, and blue bands.")
        red_idx, green_idx, blue_idx = rgb_bands
        
    elif wavelengths is not None:
        if isinstance(wavelengths, np.ndarray):
            wavelengths = wavelengths.tolist()
        
        if len(wavelengths) != spectral_data.shape[-1]:
            raise ValueError("The length of `wavelengths` must match the number of bands in `spectral_data`.")
        red_idx = np.argmin(np.abs(np.array(wavelengths) - 650))  # Closest to 650 nm (red)
        green_idx = np.argmin(np.abs(np.array(wavelengths) - 550))  # Closest to 550 nm (green)
        blue_idx = np.argmin(np.abs(np.array(wavelengths) - 450))  # Closest to 450 nm (blue)
    else:
        raise ValueError("Invalid parameters: provide either `rgb_bands` or `wavelengths`.")

    red_band = spectral_data[..., red_idx]
    red_band = normalize(red_band) * 255
    
    green_band = spectral_data[..., green_idx]
    green_band = normalize(green_band) * 255
    
    blue_band = spectral_data[..., blue_idx]
    blue_band = normalize(blue_band) * 255
    
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
    rgb_image = np.uint8(rgb_image)

    return rgb_image

Напиши документацию в стандарте numpy к этой функции на английском языке:

def simple_synthesize_rgb(band_data: list, sig_max_filt: float = None):
    '''
    Генерирует RGB-изображение из трех гиперспектральных полос.

    Функция синтезирует простое RGB-изображение из трех предоставленных полос,
    соответствующих каналам красного, зеленого и синего. Опционально, она применяет сигма 
    максимум-фильтр к каждой полосе для снижения шума перед нормализацией и масштабированием.

    Параметры
    ----------
    band_data : list of np.ndarray
        Список, содержащий ровно три 2D массива, каждый из которых представляет одну полосу данных
        для каналов красного, зеленого и синего. Каждый массив должен иметь одинаковую форму.
    sig_max_filt : float, необязательный
        Значение сигма для применения функции `sigma_maximum_filter` к каждой полосе. Если указано, оно используется 
        для снижения шума перед синтезом RGB-изображения.

    Возвращает
    ---------
    rgb_image : np.ndarray
        3D массив формы `(height, width, 3)`, представляющий синтезированное RGB-изображение.
        Выходной тип — `np.uint8` с пиксельными значениями, масштабированными в диапазон [0, 255].

    Примеры
    --------
    Генерация RGB-изображения с применением сигма максимум-фильтрации:
    
    >>> import numpy as np
    >>> from hsip.rgb.rgb import simple_synthesize_rgb
    >>> band_red = np.random.rand(100, 100)
    >>> band_green = np.random.rand(100, 100)
    >>> band_blue = np.random.rand(100, 100)
    >>> band_data = [band_red, band_green, band_blue]
    >>> rgb_image = simple_synthesize_rgb(band_data, sig_max_filt=3)
    >>> print(rgb_image.shape)
    (100, 100, 3)
    '''
    if len(band_data) != 3:
        raise ValueError("`band_data` must contain exactly three bands for red, green, and blue bands.")
    
    if sig_max_filt is not None:
        band_data = [sigma_maximum_filter(band[..., np.newaxis], sigma=3)[..., 0] for band in band_data]
        
    band_data = [normalize(band) * 255 for band in band_data]
    
    rgb_image = np.stack(band_data, axis=-1)
    rgb_image = np.uint8(rgb_image)

    return rgb_image