import pandas as pd
import numpy as np
import os
import gc
from .config import SOLAR_THRESHOLDS, SEASONS
from .features import compute_scaling_exponent

def load_and_prepare_dataframe(file_path, usecols, col_names):
    """Load dataset and create Datetime index."""
    try:
        data = np.loadtxt(file_path, skiprows=1, usecols=usecols)
        df = pd.DataFrame(data, columns=col_names)
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'minute', 'second']])
        return df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'minute', 'second'])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def classify_solar_activity(f10_7):
    """Return string label for solar activity based on F10.7 index."""
    for label, (low, high) in SOLAR_THRESHOLDS.items():
        if low <= f10_7 < high:
            return label
    return 'Unknown'

def filter_by_season(df):
    """Splits dataframe into dictionary of dataframes by season."""
    results = {}
    for season, ranges in SEASONS.items():
        mask = pd.Series(False, index=df.index)
        for (start, end) in ranges:
            mask |= (df['DOY'] >= start) & (df['DOY'] < end)
        
        subset = df[mask].copy()
        if not subset.empty:
            results[season] = subset
    return results

def calculate_gamma_for_file(df):
    """Wraps the Numba scaling exponent calculation."""
    if 'Ne' not in df.columns:
        return df
    
    ne_values = df['Ne'].values
    gamma_values = compute_scaling_exponent(ne_values)
    df['gamma2'] = gamma_values
    return df
