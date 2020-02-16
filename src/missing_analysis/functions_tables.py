"""Functions to perform the calculations needed for ``tables.py``.

"""
import numpy as np
import pandas as pd
import scipy.stats as stats


def ttest_by_column(df, dummy, equal_var):
    """Perform two-sample t-test for each column of a DataFrame, after splitting
    the observations in two groups according to a dummy variable.

    Args:
        df (pd.DataFrame): The dataframe on which to perform the t-tests.
        dummy (string): Name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.
        equal_var (bool): If True (default), perform a standard independent 2
            sample test that assumes equal population variances. If False, perform
            Welch’s t-test, which does not assume equal population variance.

    Returns:
        pd.DataFrame: A dataframe displaying, in each row,
            the two groups' mean comparison and t-test p-value for each column.

    """
    df1 = df[df[dummy] == 1].drop(dummy, axis=1)
    df0 = df[df[dummy] == 0].drop(dummy, axis=1)
    ttest_pvalue = pd.Series(
        stats.ttest_ind(
            df0.values.reshape(df0.shape),
            df1.values.reshape(df1.shape),
            axis=0,
            nan_policy="omit",
            equal_var=equal_var,
        ).pvalue,
        index=df0.columns,
    ).rename("pvalue")
    mean = df.groupby([dummy]).mean().T
    ttest_df = pd.concat(
        [mean[0].rename("mean0"), mean[1].rename("mean1"), ttest_pvalue], axis=1
    )
    return ttest_df


def compute_sample_sizes(df):
    """Compute sample sizes for control and treatment groups.

    Args:
        df (pd.DataFrame): The dataset of interest.

    Returns:
        pd.Series: Sample size for control and for treatment.

    """
    sample_size = []
    for dummy in (0, 1):
        size = pd.Series(df.groupby("treatment").size()[dummy], index=["sample_size"])
        sample_size.append(size)
    return sample_size


def levene_by_column(df, dummy):
    """Perform Levene's test for equality of variances for each column of a
    DataFrame, after splitting the observations in two groups according to a
    dummy variable.

    Args:
        df (pd.DataFrame): The dataframe on which to perform the test.
        dummy (string): Name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.

    Returns:
        pd.DataFrame: A dataframe displaying, in each row,
            the test statistic and p-value for the test for each column.

    """
    df1 = df[df[dummy] == 1].drop(dummy, axis=1)
    df0 = df[df[dummy] == 0].drop(dummy, axis=1)
    levene_outcome = []
    for col in df1.columns:
        levene_outcome.append(stats.levene(df0[col].dropna(), df1[col].dropna()))
    levene_df = pd.DataFrame(
        levene_outcome, index=df1.columns, columns=["levene_stat", "pvalue"]
    )
    return levene_df


def chisquare_by_column(df, dummy):
    """Perform chi-square t-test for each column of a DataFrame, after splitting
    the observations in two groups according to a dummy variable.

    Args:
        df (pd.DataFrame): The dataframe on which to perform the t-tests.
        dummy (string): Name of *df* column (e.g. "Treatment"). Must be a dummy
            variable.

    Returns:
        pd.DataFrame: A dataframe displaying, in each row,
            the chi-square test statistic and p-value for each column.

    """
    chisq = []
    col_index = df.drop([dummy], axis=1).columns
    for col in col_index:
        contingency_table = pd.crosstab(df[dummy], df[col], margins=True)
        f_obs = np.array(
            [
                contingency_table.iloc[0][0:2].values,
                contingency_table.iloc[1][0:2].values,
            ]
        )
        chisq.append(list(stats.chi2_contingency(f_obs)[0:2]))
    chisq_df = pd.DataFrame(chisq, index=col_index, columns=["chisq", "pvalue"])
    return chisq_df


def create_quantile_dummy(df, variable, quantile, lower=True):
    """Create dummy variable from continuous one, according to the specified
    quantile.

    Args:
        df (pd.DataFrame): Original dataframe.
        variable (string): Original (continuous) variable.
        quantile (real number or list of length 2): Quantile of the original
            continuous variable. If a real number is given, the dummy variable will
            be 1 for values of *variable* lower/higher than *quantile* and 0 elsewhere.
            If a list of length 2 is given, the dummy variable will be 1 for
            values of *variable* between the two bounds and 0 elsewhere.
        lower (boolean): Indicates whether the dummy variable is 1 for values
            of *variable* lower or higher than *quantile*. Default is True.

    Returns:
        pd.Series: Dummy variable with same index as *variable*

    Raises:
        ValueError: If *quantile* is not a real number or a list on length 2.

    """
    if isinstance(quantile, float) or isinstance(quantile, int):
        if lower is False:
            quantile_values = np.where(
                np.isnan(df[variable]),
                np.nan,
                np.where(df[variable] >= df[variable].quantile(quantile), 1, 0),
            )
        else:
            quantile_values = np.where(
                np.isnan(df[variable]),
                np.nan,
                np.where(df[variable] <= df[variable].quantile(quantile), 1, 0),
            )
    elif len(quantile) == 2:
        quantile_values = np.where(
            np.isnan(df[variable]),
            np.nan,
            np.where(
                (df[variable] > df[variable].quantile(quantile[0]))
                & (df[variable] < df[variable].quantile(quantile[1])),
                1,
                0,
            ),
        )
    else:
        raise ValueError("There must be either one or two (numeric) bounds")
    return pd.Series(quantile_values, index=df[variable].index)
