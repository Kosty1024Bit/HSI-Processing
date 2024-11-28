import numpy as np
from tqdm import tqdm

def normalize(array: np.ndarray):
    '''
    Normalizes a given NumPy array to the range [0, 1].
    The function scales the input array such that the minimum value in the array becomes 0
    and the maximum value becomes 1. The formula used is:

    .. math::
        normalized = {{value - min}} / {{max - min}}

    Parameters
    ----------
    array : np.ndarray
        A NumPy array to be normalized. The array can be of any shape or type that supports
        element-wise arithmetic operations.

    Returns
    -------
    np.ndarray
        A NumPy array of the same shape as the input, with values normalized to the range [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> array = np.array([1, 2, 3, 4, 5])
    >>> normalized_array = normalize(array)
    >>> print(normalized_array)
    [0.   0.25 0.5  0.75 1.  ]
    
    >>> array = np.array([[1, 2], [3, 4], [5, 6]])
    >>> normalized_array = normalize(array)
    >>> print(normalized_array)
    [[0.   0.2 ]
     [0.4  0.6 ]
     [0.8  1.  ]]
    '''
    
    return (array - array.min()) / (array.max() - array.min())


def rayleigh_scattering(spectral_data: np.ndarray, inplace=False, verbose=True):
    '''
    Compute the Rayleigh scattering signature for a spectral dataset.
    
    Parameters
    ----------
    spectral_data : np.ndarray
        A multi-dimensional NumPy array where the last dimension represents spectral bands.
    inplace : bool, default=False
        If True, the function modifies `spectral_data` by subtracting the computed Rayleigh signature.
    verbose : bool, verbose=True
        If True, displays a progress bar.

    Returns
    -------
    np.ndarray:
        A 1D array containing the Rayleigh scattering signature for each spectral band.
    '''
    
    if not isinstance(spectral_data, np.ndarray):
        raise TypeError('spectral_data must be an np.ndarray')
        
    bands = spectral_data.shape[-1]
    rayleigh_offsets = np.zeros(shape=bands, dtype=float)
    for i in tqdm(range(bands), disable=not verbose):
        layer = spectral_data[..., i]
        rayleigh_offsets[i] = layer[layer > 0].min()
    
    if inplace:
        spectral_data -= rayleigh_offsets
        spectral_data[spectral_data < 0] = 0
        
    return rayleigh_offsets


def sigma_maximum_filter(spectral_data: np.ndarray, sigma: float = 3, thresholds: np.ndarray = None):
    '''
    Applies a sigma-based maximum filter to the input spectral data, capping values 
    based on a calculated threshold of mean + sigma * standard deviation.

    Parameters
    ----------
    spectral_data : np.ndarray
        The input spectral data array. It is expected to have a shape where 
        the last dimension corresponds to the spectral bands.
    sigma : float, optional
        The multiplier for the standard deviation used in calculating the threshold, 
        by default 3.
    thresholds : np.ndarray, optional
        An array to store the calculated thresholds. If provided, it must have a shape 
        matching the last dimension of `spectral_data`. If `None`, thresholds are 
        computed and returned internally, by default None.

    Returns
    -------
    np.ndarray
        The filtered spectral data with values capped by the calculated thresholds.

    Notes
    -----
    - If `thresholds` is provided, it will be updated in-place with the calculated 
      threshold values.
    - The filtering is applied to all bands of the spectral data independently.

    Examples
    --------
    Apply a sigma filter to a 3D spectral dataset:
    
    >>> import numpy as np
    >>> data = np.random.rand(100, 100, 10) * 10  # Example spectral data
    >>> thresholds = np.zeros(data.shape[-1:], dtype=np.float32)
    >>> result_with_thresholds = sigma_maximum_filter(data, sigma=2, thresholds=thresholds)
    >>> thresholds  # Updated thresholds
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