"""Functions called in ``itt_analysis.py`` and ``missing_mechanism_analysis.py``."""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary_functions import chisquare_by_column
from src.auxiliary.auxiliary_functions import create_quantile_dummy
from src.auxiliary.auxiliary_functions import generate_regression_output
from src.auxiliary.auxiliary_functions import levene_by_column
from src.auxiliary.auxiliary_functions import ttest_by_column


def create_chisq_dataframe(name_df):
    """Check for missing values mechanism of ``gate_final.csv`` via chi-square
    two-sample tests and save results to ``welch_df.csv``.

    """
    df_chisq = name_df.drop(
        ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
    )
    # Create dummy variables from continuous variables
    age = create_quantile_dummy(df_chisq, "age")
    grade = create_quantile_dummy(df_chisq, "grade")
    risk = create_quantile_dummy(df_chisq, "risk_tolerance_std", median=True)
    autonomy = create_quantile_dummy(df_chisq, "autonomy_std", median=True)
    df_chisq = pd.concat([df_chisq, age, grade, risk, autonomy], axis=1)
    df_chisq = df_chisq.drop(
        ["age", "grade", "autonomy_std", "risk_tolerance_std"], axis=1
    )
    chisq_cov = chisquare_by_column(df_chisq, "missing_cov")
    chisq_out = chisquare_by_column(df_chisq, "missing_out")
    chisq_df = pd.concat(
        [chisq_cov, chisq_out],
        axis=1,
        keys=["Covariates", "Outcome of interest"],
        sort=False,
    )
    chisq_df.to_csv(ppj("OUT_ANALYSIS", "chisq_df.csv"))


def create_integrity_dataframe(name_df):
    """Verify randomization integrity of ``gate_final.csv` at application and
    follow-up wave 2 via Student's t-tests and save results to ``integrity_df.csv``.

    """
    df_app = name_df.drop(
        ["gateid", "hhincome", "hhincome_w2", "site", "missing_cov", "missing_out"],
        axis=1,
    )
    # Restrict follow-up wave 2 to individuals who stated their household income
    df_wave2 = df_app[name_df["missing_out"] == 0]
    app = ttest_by_column(df_app, "treatment", equal_var=True)
    wave2 = ttest_by_column(df_wave2, "treatment", equal_var=True)
    # Compute sample sizes for treatment and control
    sample_sizes = []
    for dataframe in (df_app, df_wave2):
        for dummy in (1, 0):
            sample_size = pd.Series(
                dataframe.groupby("treatment").size()[dummy], index=["sample_size"]
            )
            sample_sizes.append(sample_size)
    integrity_df = {
        ("Baseline", "Treatment"): app["mean1"].append(sample_sizes[0]),
        ("Baseline", "Control"): app["mean0"].append(sample_sizes[1]),
        ("Baseline", "p-value"): app["p-value"],
        ("Follow-up wave 2", "Treatment"): wave2["mean1"].append(sample_sizes[2]),
        ("Follow-up wave 2", "Control"): wave2["mean0"].append(sample_sizes[3]),
        ("Follow-up wave 2", "p-value"): wave2["p-value"],
    }
    integrity_df = pd.DataFrame.from_dict(integrity_df)
    integrity_df.to_csv(ppj("OUT_ANALYSIS", "integrity_df.csv"))


def create_levene_dataframe(name_df):
    """Perform paired Levene's test for equal variances on ``gate_final.csv``
    and save results to ``levene_df.csv``.

    """
    gate_levene = name_df.drop(
        ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
    )
    levene_cov = levene_by_column(gate_levene, "missing_cov")
    levene_out = levene_by_column(gate_levene, "missing_out")
    levene_df = pd.concat(
        [levene_cov, levene_out],
        axis=1,
        keys=["Covariates", "Outcome of interest"],
        sort=False,
    )
    levene_df.to_csv(ppj("OUT_ANALYSIS", "levene_df.csv"))


def create_logistic_dataframe(name_df):
    """Check for missing values mechanism of dataset ``gate_final.csv`` via logistic
    regression and save results to ``logistic_df.csv``.

    """
    gate_logistic = name_df.drop(
        [
            "gateid",
            "hhincome",
            "hhincome_w2",
            "completed_w2",
            "site",
            "philadelphia",
            "white",
            "hhincome_50_74k",
            "worked_for_relatives_friends_se",
        ],
        axis=1,
    )
    results_cov = generate_regression_output(
        gate_logistic.drop("missing_out", axis=1), "missing_cov", type="Logit"
    )
    results_out = generate_regression_output(gate_logistic, "missing_out", type="Logit")
    logistic_df = pd.concat(
        [results_cov, results_out],
        axis=1,
        keys=["Missing in covariates", "Missing in outcome"],
        sort=False,
    )
    logistic_df.to_csv(ppj("OUT_ANALYSIS", "logistic_df.csv"))


def create_welch_dataframe(name_df):
    """Check for missing values mechanism of ``gate_final.csv`` via
    Welch's t-tests and save results to ``welch_df.csv``.
    """
    gate_welch = name_df.drop(
        ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
    )
    ttest_cov = ttest_by_column(gate_welch, "missing_cov", equal_var=False)
    ttest_out = ttest_by_column(gate_welch, "missing_out", equal_var=False)
    welch_df = pd.concat(
        [ttest_cov, ttest_out],
        axis=1,
        keys=["Covariates", "Outcome of interest"],
        sort=False,
    )
    welch_df.to_csv(ppj("OUT_ANALYSIS", "welch_df.csv"))


def itt_analysis_with_controls(name_df, in_dir):
    """Perform OLS regression, with controls, to estimate ITT on a version of
    ``gate_final.csv``.

    Args:
        name_df (string): name of a version of ``gate_final.csv``.
        dir (string): the directory in which the version of ``gate_final.csv``
            is stored.

    Returns:
         Save regression results to .csv files.

    """
    gate_controls = pd.read_csv(ppj(in_dir, name_df + ".csv"))
    gate_controls = gate_controls.drop(
        [
            "gateid",
            "site",
            "completed_w2",
            "missing_cov",
            "missing_out",
            "hhincome",
            "white",
            "hhincome_50_74k",
            "philadelphia",
        ],
        axis=1,
    )
    complete_controls = generate_regression_output(
        gate_controls, "hhincome_w2", type="OLS"
    )
    complete_controls[0].to_csv(ppj("OUT_ANALYSIS", name_df + "_controls_coeff.csv"))
    complete_controls[1].to_csv(ppj("OUT_ANALYSIS", name_df + "_controls_summary.csv"))


def itt_analysis_without_controls(name_df, in_dir):
    """Perform OLS regression, without controls, to estimate ITT on a version of
    ``gate_final.csv``.

    Args:
        name_df (string): name of a version of ``gate_final.csv``.
        dir (string): the directory in which the version of ``gate_final.csv``
            is stored.

    Returns:
         Save regression results to .csv files.

    """
    gate_no_controls = pd.read_csv(ppj(in_dir, name_df + ".csv"))
    gate_no_controls = gate_no_controls[["hhincome_w2", "treatment"]].copy()
    complete_no_controls = generate_regression_output(
        gate_no_controls, "hhincome_w2", type="OLS"
    )
    complete_no_controls[0].to_csv(
        ppj("OUT_ANALYSIS", name_df + "_no_controls_coeff.csv")
    )
    complete_no_controls[1].to_csv(
        ppj("OUT_ANALYSIS", name_df + "_no_controls_summary.csv")
    )
