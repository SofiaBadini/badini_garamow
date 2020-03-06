"""Very inelegant ad-hoc functions to format the tables present in the final paper.

"""
import numpy as np
import pandas as pd


def format_as_percentage(df, subset):
    """Format slice of dataframe as percentage.

    Args:
        df (pd.DataFrame): the dataframe whose slice must be formatted.
        subset (IndexSlice): An argument to DataFrame.loc that restricts which
            elements of the dataframe must be formatted. The resulting subset of
            *df* must be a pd.DataFrame.

    Returns:
        pd.DataFrame: formatted dataframe.

    Raises:
        ValueError: if the subset of *df* is not a pd.DataFrame.

    """
    if isinstance(df.loc[subset], pd.DataFrame):
        df.loc[subset] = df.loc[subset].applymap(
            lambda x: x if pd.isnull(x) else f"{x:,.2%}"
        )
    else:
        raise ValueError("Subset of DataFrame must be a pd.DataFrame.")

    return df


def assign_stars(df, subset, correction=1):
    """ Assign stars for level of significance to subset of DataFrame.

    If required, implement the Bonferroni correction for multiple testing.

    Args:
        df (pd.Dataframe): The dataframe to be formatted.
        subset (IndexSlice): An argument to DataFrame.loc that restricts which
            elements of the dataframe must be formatted. The resulting subset of
            *df* must be either a pd.Series or a pd.DataFrame.
        correction (float): number of hypothesis. Default to 1 (no Bonferroni
            correction).

    Returns:
        pd.DataFrame: formatted dataframe.

    Raises:
        ValueError: if the subset of *df* is not a pd.Series or a pd.DataFrame.

    """
    significance_levels = np.array([0.1, 0.05, 0.01]) / correction
    pval = df.loc[subset]
    if isinstance(df.loc[subset], pd.DataFrame):
        formats = [
            pval.applymap(lambda x: f"{x:.3g}*"),
            pval.applymap(lambda x: f"{x:.3g}**"),
            pval.applymap(lambda x: f"{x:.3g}***"),
        ]
    elif isinstance(df.loc[subset], pd.Series):
        formats = [
            pval.apply(lambda x: f"{x:.3g}*"),
            pval.apply(lambda x: f"{x:.3g}**"),
            pval.apply(lambda x: f"{x:.3g}***"),
        ]
    else:
        raise ValueError(
            "Subset of DataFrame must be either pd.Series or\
            pd.DataFrame."
        )
    for significance_level, format in zip(significance_levels, formats):
        df.loc[subset] = df.loc[subset].mask(pval <= significance_level, format)

    return df
