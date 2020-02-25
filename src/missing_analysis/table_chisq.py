"""Check for missing values mechanism via chi-square two-sample tests.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_chisq.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.model_code.formatting_tables import assign_stars
from src.model_code.functions_tables import chisquare_by_column
from src.model_code.functions_tables import create_quantile_dummy
from src.model_code.pretty_index import pretty_index_dict


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
)

# Create dummy variables from continuous variables
age = create_quantile_dummy(gate_missing, "age")
grade = create_quantile_dummy(gate_missing, "grade")
risk = create_quantile_dummy(gate_missing, "risk_tolerance_std", median=True)
autonomy = create_quantile_dummy(gate_missing, "autonomy_std", median=True)
gate_missing = pd.concat([gate_missing, age, grade, risk, autonomy], axis=1)

# Drop continuous variables
gate_missing = gate_missing.drop(
    ["age", "grade", "autonomy_std", "risk_tolerance_std"], axis=1
)

# Compute t-test p-values and mean comparison
chisq_cov = chisquare_by_column(gate_missing, "missing_cov")
chisq_out = chisquare_by_column(gate_missing, "missing_out")

# Create MultiIndex DataFrame
table_chisq = pd.concat(
    [chisq_cov, chisq_out],
    axis=1,
    keys=["Covariates", "Outcome of interest"],
    sort=False,
)

# Format DataFrame
table_chisq = table_chisq.reindex(pretty_index_dict).rename(pretty_index_dict)
table_chisq = table_chisq.dropna(how="all")
idx = pd.IndexSlice
subset = idx[:, idx[:, "p-value"]]
correction = len(table_chisq) - 1
table_chisq = assign_stars(table_chisq, subset, correction)
table_chisq = table_chisq.fillna(" ")

# Save to latex table
table_chisq.to_latex(
    ppj("OUT_TABLES", "table_chisq.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
