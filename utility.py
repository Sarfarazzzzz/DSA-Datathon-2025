import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def missing_report(df, top_n=None, percent=True, sort=True):
    """
    Returns a DataFrame showing missing values per column (count or percent),
    with column names as rows and missing values as a column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - top_n (int, optional): Show only top N columns with the most missing data
    - percent (bool): If True, shows missing values as percentage; else shows raw counts
    - sort (bool): Sort descending by missing values

    Returns:
    - pd.DataFrame: Formatted report with columns as rows and one column showing missing data
    """
    if percent:
        missing = (df.isnull().sum() / len(df)) * 100
        label = 'Missing (%)'
    else:
        missing = df.isnull().sum()
        label = 'Missing Count'

    if sort:
        missing = missing.sort_values(ascending=False)
    if top_n:
        missing = missing.head(top_n)

    return missing.to_frame(name=label)


def get_binary_and_continuous_columns(df, binary_threshold=2, unique_cutoff=3):
    """
    Identifies binary and continuous numeric columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - binary_threshold (int): Number of unique values to consider as binary (default=2)
    - unique_cutoff (int): Minimum number of unique values to consider a column continuous (default=10)

    Returns:
    - dict: {
        'binary': [list of column names],
        'continuous': [list of column names]
      }
    """
    binary_cols = []
    continuous_cols = []

    for col in df.select_dtypes(include=['number']).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == binary_threshold:
            binary_cols.append(col)
        elif len(unique_vals) >= unique_cutoff:
            continuous_cols.append(col)

    return {
        'binary': binary_cols,
        'continuous': continuous_cols
    }

import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxplots(df, columns, figsize=(12, 6), rotation=45):
    """
    Plots boxplots for a list of numeric columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Your data
    - columns (list): List of column names to plot
    - figsize (tuple): Size of the plot canvas
    - rotation (int): Angle of x-axis labels
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(figsize[0] * num_cols, figsize[1]))

    if num_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.boxplot(data=df, y=col, ax=ax)
        ax.set_title(col)
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=rotation)

    plt.tight_layout()
    plt.show()
