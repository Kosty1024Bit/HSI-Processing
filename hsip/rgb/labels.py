import numpy as np
from hsip.rgb.colors import colors_set

def labels_to_rgb(labels: np.ndarray, RGB_image: np.ndarray = None):
    '''
    Converts a label mask into an RGB image representation.

    This function generates an RGB image where each unique label in the input `labels` array is 
    assigned a specific color. If an `RGB_image` is provided, the color for each label is 
    determined by the mean color of the corresponding regions in the `RGB_image`. If not, 
    predefined colors from a global `colors_set` are used. Warning: `colors_set` - has exactly
    17 colors in the set, if there are more unique labels, the colors for some will be repeated!

    Parameters
    ----------
    labels : np.ndarray
        A 2D or 3D array representing the label mask. Each unique value corresponds to a different class.
    RGB_image : np.ndarray, optional
        A 3D array of shape `(height, width, 3)` representing an existing RGB image. If provided, 
        the color for each label is calculated as the mean RGB value of the regions corresponding to that label.

    Returns
    -------
    np.ndarray
        A 3D RGB image of shape `(height, width, 3)` with colors assigned to each label.

    Notes
    -----
    - If `RGB_image` is not provided, the colors are assigned from a predefined `colors_set` array.
    - This function is useful for visualizing label masks in a colorful, human-readable format.

    Examples
    --------
    Example 1: Using a predefined color set:
    >>> from hsip.rgb.labels import labels_to_rgb
    >>> labels = np.array([[0, 0, 1],
    ...                    [1, 2, 2]])
    >>> rgb_image = labels_to_rgb(labels)
    >>> print(rgb_image.shape)
    (2, 3, 3)

    Example 2: Using an existing RGB image to compute colors:
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