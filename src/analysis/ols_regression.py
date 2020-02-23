"""Perform OLS regressions to estimate ITT on original dataset (complete-case
analysis), with and without controls.

"""
import pandas as pd
import statsmodels.api as sm

from bld.project_paths import project_paths_join as ppj


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# OLS without controls
y = gate_final["hhincome_w2"].values
x = sm.add_constant(gate_final["treatment"].values)
model = sm.OLS(y, x, missing="drop").fit()
model.get_robustcov_results().summary()

# OLS with set of controls
gate_controls = gate_final.drop(
    [
        "gateid",
        "site",
        "completed_w2",
        "missing_cov",
        "missing_out",
        "hhincome_w2",
        "hhincome",
        "white",
        "hhincome_p50_74",
        "philadelphia",
    ],
    axis=1,
)
x_controls = sm.add_constant(gate_controls.values)
model_controls = sm.OLS(y, x_controls, missing="drop").fit()
model_controls.get_robustcov_results().summary()
