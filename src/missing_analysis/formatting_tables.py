"""Functions to format tables.

"""
import numpy as np
import pandas as pd


def format_as_percentage(df, subset):
    """Format slice of dataframe as percentage.z

    Args:
        df (pd.DataFrame): the dataframe whose slice must be formatted.
        subset (IndexSlice): An argument to DataFrame.loc that restricts which
            elements of the dataframe must be formatted.

    Returns:
        pd.DataFrame: formatted original dataframe.

    """
    df.loc[subset] = df.loc[subset].applymap(
        lambda x: x if pd.isnull(x) else f"{x:,.2%}"
    )
    return df


def assign_stars(df, subset, correction=1):
    """ Assign stars for level of significance to subset of DataFrame.
    If required, implement the Bonferroni correction for multiple testing.

    Args:
        df (pd.Dataframe): The dataframe to be formatted.
        subset (IndexSlice): An argument to DataFrame.loc that restricts which
            elements of the dataframe must be formatted.
        correction (float): number of hypothesis. Default to 1 (no Bonferroni
            correction).

    Returns:
        pd.DataFrame: formatted original dataframe.

    """
    significance_levels = np.array([0.1, 0.05, 0.01]) / correction
    pval = df.loc[subset]
    formats = [
        pval.applymap(lambda x: f"{x:.3g}*"),
        pval.applymap(lambda x: f"{x:.3g}**"),
        pval.applymap(lambda x: f"{x:.3g}***"),
    ]
    for significance_level, format in zip(significance_levels, formats):
        df.loc[subset] = df.loc[subset].mask(pval <= significance_level, format)
    return df
