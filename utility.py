import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt

def missing_report(df, top_n=None, percent=True, sort=True, threshold=None):
    """
    Generates a bar plot showing missing values per column (as count or percentage)
    and prints a list of column names whose missing values exceed a given threshold or fall
    within a provided range.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - top_n (int, optional): Show only the top N columns with the most missing data.
    - percent (bool): If True, calculates missing values as a percentage; otherwise, as counts.
    - sort (bool): Sort the missing values in descending order.
    - threshold (float or tuple/list of two floats, optional): 
        If a single float, prints a list of columns with missing values above this threshold.
        If a tuple/list of two floats, prints a list of columns with missing values between the lower and upper bounds.
    
    Returns:
    - list: A list of column names that meet the threshold criteria.
    """
    if percent:
        missing = (df.isnull().sum() / len(df)) * 100
        ylabel = 'Missing (%)'
    else:
        missing = df.isnull().sum()
        ylabel = 'Missing Count'
    
    if sort:
        missing = missing.sort_values(ascending=False)
    if top_n:
        missing = missing.head(top_n)
    
    above_threshold = None
    # Determine threshold filtering
    if threshold is not None:
        if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
            lower, upper = threshold
            above_threshold = missing[(missing >= lower) & (missing <= upper)].index.tolist()
            print(f"Columns with missing values between {lower} and {upper} {ylabel}:")
        else:
            above_threshold = missing[missing > threshold].index.tolist()
            print(f"Columns with missing values above {threshold} {ylabel}:")
        print(above_threshold)
    
    # Create the bar plot
    plt.figure(figsize=(12, 8))
    missing.plot(kind='bar')
    plt.ylabel(ylabel)
    plt.title('Missing Data Report')
    plt.xlabel('Columns')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    return above_threshold




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


def get_statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing statistical analysis of the input DataFrame.
    
    The summary includes:
      - Datatype of the column
      - Mean (if numeric)
      - Median (if numeric)
      - Count of unique values
      - Count of null/NA values
      - Percentage of null/NA values
      - Standard deviation (if numeric)
      - Variance (if numeric)
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A summary DataFrame with the statistics for each column.
    """
    # Initialize a list to hold the results
    summary_list = []
    
    # Loop over each column in the DataFrame
    for col in df.columns:
        data = df[col]
        col_dtype = data.dtype
        unique_count = data.nunique(dropna=True)
        null_count = data.isna().sum()
        null_percentage = (null_count / len(data)) * 100
        
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(data):
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            var_val = data.var()
        else:
            mean_val = np.nan
            median_val = np.nan
            std_val = np.nan
            var_val = np.nan
        
        summary_list.append({
            "Column": col,
            "Datatype": col_dtype,
            "Mean": mean_val,
            "Median": median_val,
            "Unique Count": unique_count,
            "Null Count": null_count,
            "Null Percentage": null_percentage,
            "Std": std_val,
            "Variance": var_val
        })
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_list)
    # Optional: Set the column names as the index
    summary_df.set_index("Column", inplace=True)
    
    return summary_df



def apply_knn_imputer(df, columns=None, n_neighbors=5):

    """
    Applies KNN imputation to specified columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame
    - columns (list, optional): Subset of columns to impute. If None, all numeric columns are used.
    - n_neighbors (int): Number of neighbors to use for imputation

    Returns:
    - pd.DataFrame: New DataFrame with imputed values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include='number').columns.tolist()
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_copy[columns] = imputer.fit_transform(df_copy[columns])
    
    return df_copy

import pandas as pd

def smart_impute(df, cols, group_col=None, skew_threshold=1):
    """
    Imputes missing values in specified numeric columns using group-specific statistics if group_col is provided,
    otherwise performs simple imputation based on overall column skew.
    
    For each column:
      - Only numeric columns are imputed.
      - If group_col is provided, groups the DataFrame by `group_col` and fills missing values with:
          - The group's median if the column's skew > skew_threshold,
          - Otherwise, the group's mean.
      - If group_col is None, fills missing values with:
          - The column's median if its skew > skew_threshold,
          - Otherwise, the column's mean.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cols (list): List of column names to impute.
    - group_col (str, optional): Column name to group by. If not provided, performs global imputation.
    - skew_threshold (float): Threshold for skew to decide median vs. mean imputation (default 1).
    
    Returns:
    - pd.DataFrame: A new DataFrame with imputed values for numeric columns.
    """
    df_copy = df.copy()
    
    for col in cols:
        # Only process numeric columns
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        if group_col is not None:
            df_copy[col] = df_copy.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.median() if x.skew() > skew_threshold else x.mean())
            )
        else:
            if df_copy[col].skew() > skew_threshold:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
    
    return df_copy
