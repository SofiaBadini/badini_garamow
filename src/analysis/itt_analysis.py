"""Perform OLS regressions to estimate ITT on datasets stored in the "OUT_DATA"
and in the "OUT_IMPUTED_DATA" directory. Results are saved to .csv files and stored
in the "OUT_ANALYSIS" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary_functions import generate_regression_output


def itt_analysis_with_controls(name_df, dir):
    """Perform OLS regression, with controls, to estimate ITT on a version of
    ``gate_final.csv``.

    Args:
        name_df (string): name of a version of ``gate_final.csv``.
        dir (string): the directory in which the version of ``gate_final.csv``
            is stored.

    Returns:
         Save regression results to .csv files.

    """
    gate_controls = pd.read_csv(ppj(dir, name_df + ".csv"))
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


def itt_analysis_without_controls(name_df, dir):
    """Perform OLS regression, without controls, to estimate ITT on a version of
    ``gate_final.csv``.

    Args:
        name_df (string): name of a version of ``gate_final.csv``.
        dir (string): the directory in which the version of ``gate_final.csv``
            is stored.

    Returns:
         Save regression results to .csv files.

    """
    gate_no_controls = pd.read_csv(ppj(dir, name_df + ".csv"))
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


if __name__ == "__main__":
    itt_analysis_with_controls("gate_complete", "OUT_DATA")
    itt_analysis_without_controls("gate_complete", "OUT_DATA")
