import pandas as pd

def load_data(filepath, delimiter='|'):
    """
    Loads data from a CSV/Text file.
    
    Args:
        filepath (str): Path to the file.
        delimiter (str): Delimiter used in the file. Default is '|'.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
