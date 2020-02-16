"""Functions to format tables.

"""
import numpy as np
import pandas as pd


def format_as_percentage(df, row_to_format, column_to_format):
    """Format slice of dataframe as percentage.

    Args:
        df (pd.DataFrame): the dataframe whose slice must be formatted.
        row_to_format (string, list, IndexSlice object): row(s) to be formatted.
        column_to_format (string, list, IndexSlice object): column(s) to be
            formatted.

    Returns:
        pd.DataFrame: formatted original dataframe.

    """
    subset_to_format = df.loc[row_to_format, column_to_format]
    if isinstance(subset_to_format, pd.DataFrame):
        df.loc[row_to_format, column_to_format] = subset_to_format.applymap(
            lambda x: f"{x:,.2%}"
        )
    elif isinstance(subset_to_format, pd.Series):
        df.loc[row_to_format, column_to_format] = subset_to_format.apply(
            lambda x: f"{x:,.2%}"
        )
    else:
        df.loc[row_to_format, column_to_format] = f"{subset_to_format:,.2%}"
    return df


def assign_stars_to_column(df, column_to_format, correction):
    """ Assign stars for level of significance to specific column(s) of
    DataFrame, implementing the Bonferroni correction for multiple testing.

    Args:
        df (pd.Dataframe): the dataframe whose column(s) must be formatted.
        column_to_format (string, list, IndexSlice object): column(s) to be
            formatted.
        correction (float): number of hypothesis.

    Returns:
        pd.DataFrame: formatted original dataframe.

    """
    significance_levels = np.array([0.1, 0.05, 0.01]) / correction
    pval = df.loc[:, column_to_format]
    formats = [
        pval.applymap(lambda x: f"{x:.3g}*"),
        pval.applymap(lambda x: f"{x:.3g}**"),
        pval.applymap(lambda x: f"{x:.3g}***"),
    ]
    for significance_level, format in zip(significance_levels, formats):
        df.loc[:, column_to_format] = df.loc[:, column_to_format].mask(
            pval <= significance_level, format
        )
    return df
