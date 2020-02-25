"""Perform OLS regressions to estimate ITT on original dataset (complete-case
analysis), with and without controls.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_complete_withcontr0.tex``, ``table_complete_withcontr1.tex``,
``table_complete_nocontr0.tex``, and ``table_complete_nocontr1.tex`` in the
"OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.formatting_tables import assign_stars
from src.model_code.functions_tables import generate_regression_output
from src.model_code.pretty_index import pretty_index_dict

# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_withcontr = gate_final.drop(
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

# Run OLS regression with controls and create results' DataFrame
complete_withcontr = generate_regression_output(
    gate_withcontr, "hhincome_w2", type="OLS"
)

# Format DataFrame
complete_withcontr[0] = (
    complete_withcontr[0].reindex(pretty_index_dict).rename(pretty_index_dict)
)
complete_withcontr[0] = complete_withcontr[0].dropna(how="all")
subset = pd.IndexSlice[:, "p-value"]
complete_withcontr[0] = assign_stars(complete_withcontr[0], subset)

# Run OLS regression without controls and create results' DataFrame
gate_nocontr = gate_final[["hhincome_w2", "treatment"]]
complete_nocontr = generate_regression_output(gate_nocontr, "hhincome_w2", type="OLS")

# Format DataFrame
complete_nocontr[0] = (
    complete_nocontr[0].reindex(pretty_index_dict).rename(pretty_index_dict)
)
complete_nocontr[0] = complete_nocontr[0].dropna(how="all")

# Save to latex table
for i in (0, 1):
    complete_withcontr[i].to_latex(
        ppj("OUT_TABLES", "table_complete_withcontr" + str(i) + ".tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )
    complete_nocontr[i].to_latex(
        ppj("OUT_TABLES", "table_complete_nocontr" + str(i) + ".tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )
