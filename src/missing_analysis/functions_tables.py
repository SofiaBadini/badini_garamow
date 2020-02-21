"""Functions to perform the calculations needed to generate the tables.

"""
import numpy as np
import pandas as pd
import scipy.stats as stats


def ttest_by_column(df, dummy, equal_var=True):
    """Iterate two-sample t-test for each column of a DataFrame, after splitting
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
    ttest_pval = pd.Series(
        stats.ttest_ind(
            df0.values.reshape(df0.shape),
            df1.values.reshape(df1.shape),
            axis=0,
            nan_policy="omit",
            equal_var=equal_var,
        ).pvalue,
        index=df0.columns,
    )
    mean = df.groupby([dummy]).mean().T
    ttest_df = pd.concat(
        [mean[0].rename("mean0"), mean[1].rename("mean1"), ttest_pval.rename("pvalue")],
        axis=1,
    )
    return ttest_df


def compute_sample_sizes(df, dummy):
    """Split dataset in two groups according to a dummy variable and count the
    number of observations in each group.

    Args:
        df (pd.DataFrame): The dataset of interest.
        dummy (string): Name of *df* column (e.g. "Treatment"). Must be
            a dummy variable.

    Returns:
        list of pd.Series: Sample size for control and for treatment, with
            index "sample_size".

    """
    sample_size = []
    for value in (0, 1):
        size = pd.Series(df.groupby(dummy).size()[value], index=["sample_size"])
        sample_size.append(size)
    return sample_size


def levene_by_column(df, dummy):
    """Iterate Levene's test for equality of variances for each column of a
    DataFrame, after splitting the observations in two groups according to a
    dummy variable.

    Args:
        df (pd.DataFrame): The dataframe on which to perform the test.
        dummy (string): Name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.

    Returns:
        pd.DataFrame: A dataframe displaying, in each row,
            the Levene's test statistic and p-value for each column.

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
    """Iterate chi-square t-test for each column of a DataFrame, after splitting
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


def create_quantile_dummy(df, variable, median=False):
    """Create dummy variable(s) from continuous one.

    Args:
        df (pd.DataFrame): Original dataframe.
        variable (string): Original (continuous) variable.
        median (boolean): If True, the returned dummy variable is 1 for values
            of *variable* lower than the median. If False, three dummy variables
            are returned respectively for values of *variable* lower than the
            first quantile, between the first and the third quantile, and higher
            than the third quantile. Default is False.

    Returns:
        pd.Series or pd.DataFrame: Dummy variable(s) with same index as *variable*.

    """
    if median is False:
        values_p25 = np.where(
            np.isnan(df[variable]),
            np.nan,
            np.where(df[variable] <= df[variable].quantile(0.25), 1, 0),
        )
        values_p25_75 = np.where(
            np.isnan(df[variable]),
            np.nan,
            np.where(
                (df[variable] > df[variable].quantile(0.25))
                & (df[variable] < df[variable].quantile(0.75)),
                1,
                0,
            ),
        )
        values_p75 = np.where(
            np.isnan(df[variable]),
            np.nan,
            np.where(df[variable] >= df[variable].quantile(0.75), 1, 0),
        )
        dummies = np.vstack((values_p25, values_p25_75, values_p75)).T
        out = pd.DataFrame(
            dummies,
            columns=[variable + "_p25", variable + "_p25_75", variable + "_p75"],
            index=df.index,
        )
    else:
        values = np.where(
            np.isnan(df[variable]),
            np.nan,
            np.where(df[variable] <= df[variable].quantile(0.5), 1, 0),
        )
        out = pd.Series(values, name="low_" + variable, index=df.index)
    return out
