# src/ibbi/utils/info.py

"""
This module provides utility functions for displaying package information, such as
a list of available models and their performance metrics. These functions are
designed to help users explore the capabilities of the `ibbi` package and make
informed decisions about which models to use for their specific tasks.
"""

from importlib import resources

import pandas as pd


def list_models(as_df: bool = False):
    """Displays or returns a summary of available models and their key information.

    This function reads the model summary CSV file included with the package, which
    contains a comprehensive list of all available models, their tasks, and key
    performance metrics. It can either print this information to the console in a
    human-readable format or return it as a pandas DataFrame for programmatic access.

    Args:
        as_df (bool, optional): If True, the function returns the model information as a
                                pandas DataFrame. If False (the default), it prints the
                                information directly to the console.

    Returns:
        pd.DataFrame or None: If `as_df` is True, a pandas DataFrame containing the model
                              summary is returned. Otherwise, the function returns None.
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
