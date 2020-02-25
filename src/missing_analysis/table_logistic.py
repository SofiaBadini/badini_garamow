"""Check for missing mechanism via logistic regressions.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_legistic.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.formatting_tables import assign_stars
from src.model_code.functions_tables import generate_regression_output
from src.model_code.pretty_index import pretty_index_dict


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables from dataset
gate_logit = gate_final.drop(
    [
        "gateid",
        "hhincome",
        "hhincome_w2",
        "completed_w2",
        "site",
        "philadelphia",
        "white",
        "hhincome_p50_74",
        "worked_for_relatives_friends_se",
    ],
    axis=1,
)

# Compute results
results_cov = generate_regression_output(
    gate_logit.drop("missing_out", axis=1), "missing_cov", type="Logit"
)
results_out = generate_regression_output(gate_logit, "missing_out", type="Logit")

# Create MultiIndex DataFrame
table_logistic = pd.concat(
    [results_cov, results_out],
    axis=1,
    keys=["Missing in covariates", "Missing in outcome"],
    sort=False,
)

# Format DataFrame
table_logistic = table_logistic.reindex(pretty_index_dict).rename(pretty_index_dict)
table_logistic = table_logistic.dropna(how="all")
idx = pd.IndexSlice
subset = idx[:, idx[:, "p-value"]]
table_logistic = assign_stars(table_logistic, subset)
table_logistic = table_logistic.fillna(" ")

# Save to latex table
table_logistic.to_latex(
    ppj("OUT_TABLES", "table_logistic.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)
