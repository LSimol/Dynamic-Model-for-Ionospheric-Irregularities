import numpy as np
from numba import jit

@jit(nopython=True)
def linear_regression(x, y):
    """
    Perform linear regression using Numba for optimization. 
    Equivalent to np.polyfit(x,y,1).
    """
    n = len(x)
    x_mean = np.sum(x) / n
    y_mean = np.sum(y) / n

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)

    if den == 0:
        return np.nan

    return num / den

@jit(nopython=True)
def compute_scaling_exponent(signal, window_size=300, num_increments=10, min_tau=1, max_tau=40):
    """
    Optimized function using Numba for structure function scaling exponent.
    """
    num_points = len(signal)
    scaling_exponents = np.full(num_points - window_size, np.nan) 

    increments = np.logspace(np.log10(min_tau), np.log10(max_tau), num_increments).astype(np.int32)
    log_increments = np.log(increments)

    for i in range(num_points - window_size):
        window = signal[i : i + window_size]

        if np.isnan(window[0]):
            continue

        structure_function = np.zeros(len(increments))
        valid_struct = True

        for j in range(len(increments)):
            tau = increments[j]
            diffs = window[tau:] - window[:-tau]
            mean_sq_diff = np.nanmean(diffs ** 2)
            
            if mean_sq_diff <= 0:
                valid_struct = False
                break
            structure_function[j] = mean_sq_diff

        if not valid_struct:
            scaling_exponents[i] = np.nan
            continue

        log_S2 = np.log(structure_function)

        if np.any(np.isnan(log_S2)) or np.any(np.isinf(log_S2)):
            scaling_exponents[i] = np.nan
        else:
            scaling_exponents[i] = linear_regression(log_increments, log_S2)

    # Pad result to match original signal length (centering or trailing padding)
    # The original notebook padded with 150 on both sides
    pad_width = window_size // 2
    scaling_exponents_padded = np.full(num_points, np.nan)
    scaling_exponents_padded[pad_width : pad_width + len(scaling_exponents)] = scaling_exponents
    
    return scaling_exponents_paddedimport numpy as np
from numba import jit

@jit(nopython=True)
def linear_regression(x, y):
    """
    Perform linear regression using Numba for optimization. 
    Equivalent to np.polyfit(x,y,1).
    """
    n = len(x)
    x_mean = np.sum(x) / n
    y_mean = np.sum(y) / n

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)

    if den == 0:
        return np.nan

    return num / den

@jit(nopython=True)
def compute_scaling_exponent(signal, window_size=300, num_increments=10, min_tau=1, max_tau=40):
    """
    Optimized function using Numba for structure function scaling exponent.
    """
    num_points = len(signal)
    scaling_exponents = np.full(num_points - window_size, np.nan) 

    increments = np.logspace(np.log10(min_tau), np.log10(max_tau), num_increments).astype(np.int32)
    log_increments = np.log(increments)

    for i in range(num_points - window_size):
        window = signal[i : i + window_size]

        if np.isnan(window[0]):
            continue

        structure_function = np.zeros(len(increments))
        valid_struct = True

        for j in range(len(increments)):
            tau = increments[j]
            diffs = window[tau:] - window[:-tau]
            mean_sq_diff = np.nanmean(diffs ** 2)
            
            if mean_sq_diff <= 0:
                valid_struct = False
                break
            structure_function[j] = mean_sq_diff

        if not valid_struct:
            scaling_exponents[i] = np.nan
            continue

        log_S2 = np.log(structure_function)

        if np.any(np.isnan(log_S2)) or np.any(np.isinf(log_S2)):
            scaling_exponents[i] = np.nan
        else:
            scaling_exponents[i] = linear_regression(log_increments, log_S2)

    # Pad result to match original signal length (centering or trailing padding)
    # The original notebook padded with 150 on both sides
    pad_width = window_size // 2
    scaling_exponents_padded = np.full(num_points, np.nan)
    scaling_exponents_padded[pad_width : pad_width + len(scaling_exponents)] = scaling_exponents
    
    return scaling_exponents_padded
