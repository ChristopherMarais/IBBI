# src/ibbi/utils/info.py

"""
Utility functions for displaying package information.
"""

from importlib import resources

import pandas as pd


def list_models(as_df: bool = False):
    """
    Displays available models and their key information.

    Reads the model summary CSV included with the package and prints it.
    Can also return the data as a pandas DataFrame.

    Args:
        as_df (bool): If True, returns the model information as a pandas DataFrame.
                      If False (default), prints the information to the console.

    Returns:
        pd.DataFrame or None: A DataFrame if as_df is True, otherwise None.
    """
    try:
        # Find the path to the data file within the package
        with resources.files("ibbi.data").joinpath("ibbi_model_summary.csv").open("r") as f:
            df = pd.read_csv(f)

        if as_df:
            return df
        else:
            print("Available IBBI Models:")
            print(df.to_string())

    except FileNotFoundError:
        print("Error: Model summary file not found.")
        return None
