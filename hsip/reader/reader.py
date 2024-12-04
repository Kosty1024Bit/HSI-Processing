import numpy as np

import spectral.io.spyfile as spyfile
import spectral.io.aviris as aviris
import spectral.io.envi as envi
import spectral.io.erdas as erdas

import tifffile as tfl


def open_SpyFile(path_lan: str):
    '''
    Открывает гиперспектральное изображение в формате SpyFile.

    Параметры
    ---------
    path_lan : str
        Путь к LAN-файлу, который необходимо открыть.

    Возвращаемые значения
    ---------------------
    SpectralLibrary
        Объект `SpyFile`, содержащий гиперспектральные данные.

    Примеры
    -------
    >>> from hsip.reader.reader import open_SpyFile
    >>> hsi = open_SpyFile("example.lan")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    
    hsi = spyfile.open(path_lan)
    return hsi
    
    
def open_ENVI(path_hdr: str, path_img: str):
    '''
    Открывает гиперспектральное изображение в формате ENVI и конвертирует его в массив NumPy.

    Параметры
    ---------
    path_hdr : str
        Путь к заголовочному файлу ENVI (.hdr).
    path_img : str
        Путь к изображению ENVI (.img).

    Возвращаемые значения
    ---------------------
    np.ndarray
        Массив NumPy с гиперспектральными данными, с типом данных `dtype=float`.

    Примеры
    -------
    >>> from hsip.reader.reader import open_ENVI
    >>> hsi = open_ENVI("example.hdr", "example.img")
    >>> print(hsi.shape)
    (100, 100, 224)  # Примерные размеры
    '''
    
    hsi_envi = envi.open(path_hdr, path_img)
    hsi = np.array(hsi_envi.open_memmap(writble=True), dtype=float)
    return hsi

    
def open_AVIRIS(path_rfl: str, path_spc: str):
    '''
    Открывает гиперспектральное изображение в формате AVIRIS.

    Параметры
    ---------
    path_rfl : str
        Путь к файлу данных отражения AVIRIS (.rfl).
    path_spc : str
        Путь к файлу спектральной калибровки AVIRIS (.spc).

    Возвращаемые значения
    ---------------------
    SpectralLibrary
        Объект `SpyFile`, содержащий гиперспектральные данные AVIRIS.

    Примеры
    -------
    >>> from hsip.reader.reader import open_AVIRIS
    >>> hsi = open_AVIRIS("example.rfl", "example.spc")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    hsi = aviris.open(path_rfl, path_spc)
    return hsi


def open_ERDAS(path: str):
    '''
    Открывает гиперспектральное изображение в формате ERDAS Imagine.

    Параметры
    ---------
    path : str
        Путь к файлу ERDAS Imagine, который необходимо открыть.

    Возвращаемые значения
    ---------------------
    SpectralLibrary
        Объект `SpyFile`, содержащий гиперспектральные данные.

    Примеры
    -------
    >>> from hsip.reader.reader import open_ERDAS
    >>> hsi = open_ERDAS("example.img")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    
    hsi = erdas.open(path)
    return hsi


def open_TIF(path_tif: str):
    '''
    Открывает гиперспектральное изображение в формате GeoTIFF.

    Параметры
    ---------
    path_tif : str
        Путь к файлу GeoTIFF, который необходимо открыть.

    Возвращаемые значения
    ---------------------
    np.ndarray
        Массив NumPy, содержащий гиперспектральные данные.

    Примеры
    -------
    >>> from hsip.reader.reader import open_TIF
    >>> hsi = open_TIF("example.tif")
    >>> print(hsi.shape)
    (100, 100, 224)
    '''
    
    hsi_tif = tfl.TiffFile(path_tif)
    hsi = hsi_tif.asarray().copy()
    return hsi