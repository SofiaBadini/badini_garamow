"""Perform OLS regressions to estimate ITT on original dataset (complete-case
analysis), with controls.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_ols_controls.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars
from src.missing_analysis.functions_tables import generate_regression_output
from src.missing_analysis.pretty_index import pretty_index_dict

# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# OLS without controls, to be moved to different place
# gate_no_controls = gate_final[["hhincome_w2", "treatment"]]
# model_no_controls = generate_regression_output(
#    gate_no_controls, "hhincome_w2", type="OLS"
#    )

# Drop redundant variables
gate_controls = gate_final.drop(
    [
        "gateid",
        "site",
        "completed_w2",
        "missing_cov",
        "missing_out",
        "hhincome",
        "white",
        "hhincome_p50_74",
        "philadelphia",
    ],
    axis=1,
)

# Run OLS regression
table_ols_controls = generate_regression_output(
    gate_controls, "hhincome_w2", type="OLS"
)

# Format DataFrame
table_ols_controls = table_ols_controls.rename(
    columns=({"coef": "Coeff.", "std err": "Std. Error", "P>|t|": "p-value"})
)
table_ols_controls = table_ols_controls.reindex(pretty_index_dict).rename(
    pretty_index_dict
)
table_ols_controls = table_ols_controls.dropna(how="all")
table_ols_controls = table_ols_controls[["Coeff.", "Std. Error", "p-value"]].copy()
subset = pd.IndexSlice[:, "p-value"]
table_ols_controls = assign_stars(table_ols_controls, subset)

# Save to latex table
table_ols_controls.to_latex(
    ppj("OUT_TABLES", "table_ols_controls.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)
