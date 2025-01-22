# -*- coding: utf-8 -*-
"""
Purpose: Generate new features.

Content:
- Functions to derive new columns.
- Feature transformations (e.g., time-series features).
"""

import numpy as np
import pandas as pd

def create_lag_features(data, column_name, lags):
    """
    Create lag features for a specific column in the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        column_name (str): Name of the column for which lag features will be created.
        lags (list of int): List of lag intervals.

    Returns:
        pd.DataFrame: Dataset with added lag features.
    """
    for lag in lags:
        data[f"{column_name}_lag_{lag}"] = data[column_name].shift(lag)
    return data


def create_rolling_features(data, column_name, window_sizes, aggregation_funcs):
    """
    Create rolling window features for a specific column in the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        column_name (str): Name of the column for which rolling features will be created.
        window_sizes (list of int): List of rolling window sizes.
        aggregation_funcs (list of str): List of aggregation functions (e.g., 'mean', 'std', 'max').

    Returns:
        pd.DataFrame: Dataset with added rolling features.
    """
    for window in window_sizes:
        for func in aggregation_funcs:
            data[f"{column_name}_rolling_{func}_{window}"] = (
                data[column_name]
                .rolling(window=window, min_periods=1)
                .agg(func)
            )
    return data


def create_time_features(data, datetime_column):
    """
    Create time-based features from a datetime column.

    Args:
        data (pd.DataFrame): Input dataset.
        datetime_column (str): Name of the datetime column.

    Returns:
        pd.DataFrame: Dataset with added time features (year, month, day, hour, etc.).
    """
    data[datetime_column] = pd.to_datetime(data[datetime_column])
    data["year"] = data[datetime_column].dt.year
    data["month"] = data[datetime_column].dt.month
    data["day"] = data[datetime_column].dt.day
    data["weekday"] = data[datetime_column].dt.weekday
    data["hour"] = data[datetime_column].dt.hour
    return data


def generate_polynomial_features(data, column_name, degree):
    """
    Generate polynomial features for a specific column.

    Args:
        data (pd.DataFrame): Input dataset.
        column_name (str): Name of the column for which polynomial features will be created.
        degree (int): Degree of the polynomial features.

    Returns:
        pd.DataFrame: Dataset with added polynomial features.
    """
    for d in range(2, degree + 1):
        data[f"{column_name}_poly_{d}"] = data[column_name] ** d
    return data



import pandas as pd
from feature_generation import (
    create_lag_features,
    create_rolling_features,
    create_time_features,
    generate_polynomial_features,
)

# Example dataset
data = pd.DataFrame({
    "datetime": pd.date_range(start="2023-01-01", periods=10, freq="D"),
    "value": np.random.rand(10)
})

# Create lag features
data = create_lag_features(data, column_name="value", lags=[1, 2, 3])

# Create rolling features
data = create_rolling_features(data, column_name="value", window_sizes=[2, 3], aggregation_funcs=["mean", "std"])

# Create time-based features
data = create_time_features(data, datetime_column="datetime")

# Generate polynomial features
data = generate_polynomial_features(data, column_name="value", degree=3)

print(data)
