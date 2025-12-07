import pandas as pd
from src.loader import load_data

def test_load_data_file_not_found():
    """
    Test that load_data returns None when file is missing.
    """
    result = load_data("non_existent_file.csv")
    assert result is None
