import pandas as pd

def load_data(path):
    """Load base churn dataset"""
    return pd.read_csv(path)
