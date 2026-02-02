"""
Preprocessing utilities for scaling/unscaling data.
"""

import numpy as np


def inverse_scale_minmax(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Inverse min-max scaling: convert [0, 1] back to [min_val, max_val].

    Args:
        data: Scaled data in [0, 1] range
        min_val: Original minimum value
        max_val: Original maximum value

    Returns:
        Data in original scale [min_val, max_val]
    """
    return data * (max_val - min_val) + min_val


def scale_minmax(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Min-max scaling: convert [min_val, max_val] to [0, 1].

    Args:
        data: Data in original scale
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling

    Returns:
        Scaled data in [0, 1] range
    """
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / range_val
