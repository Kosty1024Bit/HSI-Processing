import numpy as np

import spectral.io.spyfile as spyfile
import spectral.io.aviris as aviris
import spectral.io.envi as envi
import spectral.io.erdas as erdas

import tifffile as tfl


def open_SpyFile(path_lan: str):
    '''
    Opens a hyperspectral image in LAN format.

    Parameters
    ----------
    path_lan : str
        Path to the LAN file to be opened.

    Returns
    -------
    SpectralLibrary
        A `SpyFile` object containing the hyperspectral data.

    Examples
    --------
    >>> from hsip.reader.reader import open_SpyFile
    >>> hsi = open_SpyFile("example.lan")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    
    hsi = spyfile.open(path_lan)
    return hsi
    
    
def open_ENVI(path_hdr: str, path_img: str):
    '''
    Opens a hyperspectral image in ENVI format and converts it to a NumPy array.

    Parameters
    ----------
    path_hdr : str
        Path to the ENVI header file (.hdr).
    path_img : str
        Path to the ENVI image file (.img).

    Returns
    -------
    np.ndarray
        A NumPy array of the hyperspectral data, with `dtype=float`.

    Notes
    -----
    - The ENVI image is opened as a memory-mapped array and then loaded fully into memory as a float array.

    Examples
    --------
    >>> from hsip.reader.reader import open_ENVI
    >>> hsi = open_ENVI("example.hdr", "example.img")
    >>> print(hsi.shape)
    (100, 100, 224)  # Example dimensions
    '''
    
    hsi_envi = envi.open(path_hdr, path_img)
    hsi = np.array(hsi_envi.open_memmap(writble=True), dtype=float)
    return hsi

    
def open_AVIRIS(path_rfl: str, path_spc: str):
    '''
    Opens a hyperspectral image in AVIRIS format.

    Parameters
    ----------
    path_rfl : str
        Path to the AVIRIS reflectance data file (.rfl).
    path_spc : str
        Path to the AVIRIS spectral calibration file (.spc).

    Returns
    -------
    SpectralLibrary
        A `SpyFile` object containing the AVIRIS hyperspectral data.

    Examples
    --------
    >>> from hsip.reader.reader import open_AVIRIS
    >>> hsi = open_AVIRIS("example.rfl", "example.spc")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    hsi = aviris.open(path_rfl, path_spc)
    return hsi


def open_ERDAS(path: str):
    '''
    Opens a hyperspectral image in ERDAS Imagine format.

    Parameters
    ----------
    path : str
        Path to the ERDAS Imagine file to be opened.

    Returns
    -------
    SpectralLibrary
        A `SpyFile` object containing the hyperspectral data.

    Examples
    --------
   >>> from hsip.reader.reader import open_ERDAS
    >>> hsi = open_ERDAS("example.img")
    >>> print(hsi)
    SpyFile: [shape=(100, 100, 224), dtype=float32]
    '''
    
    hsi = erdas.open(path)
    return hsi


def open_TIF(path_tif: str):
    '''
    Opens a hyperspectral image in GeoTIFF format.

    Parameters
    ----------
    path_tif : str
        Path to the GeoTIFF file to be opened.

    Returns
    -------
    np.ndarray
        A NumPy array containing the hyperspectral data.

    Notes
    -----
    - The GeoTIFF data is loaded into memory using the `tifffile` library and copied as a NumPy array.

    Examples
    --------
    >>> >>> from hsip.reader.reader import open_TIF
    >>> hsi = open_TIF("example.tif")
    >>> print(hsi.shape)
    (100, 100, 224)  # Example dimensions
    '''
    
    hsi_tif = tfl.TiffFile(path_tif)
    hsi = hsi_tif.asarray().copy()
    return hsi