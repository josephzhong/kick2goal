import numpy as np


def interquartile_mean(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    quartile = np.percentile(arr, 25)
    quartile_3rd = np.percentile(arr, 75)
    index = np.where((arr >= quartile) & (arr <= quartile_3rd))
    return np.mean(arr[index])
