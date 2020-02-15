import numpy as np
import pandas as pd
import scipy.stats as stats


def ttest_by_column(df, dummy, equal_var):
    """Perform two-sample t-test for each column of a DataFrame, after splitting
    the observations in two groups according to a dummy variable.

    Args:
        df (pd.DataFrame): the dataframe on which to perform the t-tests
        dummy (string): name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.
        equal_var (bool): If True (default), perform a standard independent 2
            sample test that assumes equal population variances. If False,
            perform Welch’s t-test, which does not assume equal population
            variance.
    Returns:
        ttest_df (pd.DataFrame): a dataframe displaying, in each row,
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


def levene_by_column(df, dummy):
    """Perform Levene's test for equality of variances for each column of a
    DataFrame, after splitting the observations in two groups according to a
    dummy variable.

    Args:
        df (pd.DataFrame): the dataframe on which to perform the test
        dummy (string): name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.
    Returns:
        levene_df (pd.DataFrame): a dataframe displaying, in each row,
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
        df (pd.DataFrame): the dataframe on which to perform the t-tests
        dummy (string): name of *df* column (e.g. "Treatment"). Must represent
            a dummy variable.
    Returns:
        chisq_df (pd.DataFrame): a dataframe displaying, in each row,
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
