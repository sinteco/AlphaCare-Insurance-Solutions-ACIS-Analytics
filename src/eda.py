import pandas as pd

def check_missing_values(df):
    """
    Checks for missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.Series: Count of missing values per column.
    """
    return df.isnull().sum()

def get_descriptive_stats(df):
    """
    Returns descriptive statistics.
    """
    return df.describe()
