import numpy as np
import pandas as pd


# Maybe add link to https://docs.python.org/2/library/string.html#formatspec
# in documentation
def format_table(df, rows_to_format, columns_to_format, format_string):
    """Format specific slice of a DataFrame with two-levels MultiIndex columns.

    Args:
        df (pd.DataFrame): the dataframe whose slice must be formatted
        rows_to_format (string): name of row(s) to be formatted
        columns_to_format (string): name of second-level column(s) to be formatted
        format_string (string): format string defining how values are presented
    Returns:
        formatted original dataframe

    """
    idx = pd.IndexSlice
    subset_to_format = df.loc[rows_to_format, idx[:, (columns_to_format)]]
    if isinstance(df.loc[rows_to_format], pd.Series):
        df.loc[rows_to_format, idx[:, (columns_to_format)]] = subset_to_format.apply(
            lambda x: format_string.format(x)
        )
    else:
        df.loc[rows_to_format, idx[:, (columns_to_format)]] = subset_to_format.applymap(
            lambda x: format_string.format(x)
        )
    return df


def assign_stars(df, column_to_format, correction):
    """ Assign stars for level of significance to specific second-level
    column(s) of DataFrame with two-levels MultiIndex columns, implementing
    the Bonferroni correction for multiple testing.

    Args:
        df (pd.Dataframe): the dataframe whose column(s) must be formatted
        column_to_format (string): name of second-level column(s) to be formatted
        correction (float): number of hypothesis
    Returns:
        formatted original dataframe

    """
    idx = pd.IndexSlice
    significance_levels = np.array([0.1, 0.05, 0.01]) / correction
    pval = df.loc[:, idx[:, (column_to_format)]]
    formats = [
        pval.applymap(lambda x: f"{x:.3g}*"),
        pval.applymap(lambda x: f"{x:.3g}**"),
        pval.applymap(lambda x: f"{x:.3g}***"),
    ]
    for significance_level, format in zip(significance_levels, formats):
        df.loc[:, idx[:, (column_to_format)]] = df.loc[
            :, idx[:, (column_to_format)]
        ].mask(pval <= significance_level, format)
    return df
