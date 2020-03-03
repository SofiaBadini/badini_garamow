"""Perform OLS regressions to estimate ITT in ``gate_final.csv``, stored in the
"OUT_DATA" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.auxiliary_functions import generate_regression_output


gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))


def create_complete_case_controls_dataframe():
    """Perform ITT complete-case analysis, with controls, on ``gate_final.csv``,
    and save results to "complete_controls_coeff.csv", and "complete_controls_summary.csv".

    """
    gate_controls = gate_final.drop(
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
    complete_controls[0].to_csv(ppj("OUT_ANALYSIS", "complete_controls_coeff_df.csv"))
    complete_controls[1].to_csv(ppj("OUT_ANALYSIS", "complete_controls_summary_df.csv"))


def create_complete_case_no_controls_dataframe():
    """Perform ITT complete-case analysis, without controls, on ``gate_final.csv``,
    and save results to "complete_no_controls_coeff.csv", and
    "complete_no_controls_summary.csv".

    """
    gate_no_controls = gate_final[["hhincome_w2", "treatment"]]
    complete_no_controls = generate_regression_output(
        gate_no_controls, "hhincome_w2", type="OLS"
    )
    complete_no_controls[0].to_csv(
        ppj("OUT_ANALYSIS", "complete_no_controls_coeff_df.csv")
    )
    complete_no_controls[1].to_csv(
        ppj("OUT_ANALYSIS", "complete_no_controls_summary_df.csv")
    )


create_complete_case_no_controls_dataframe()
create_complete_case_controls_dataframe()
