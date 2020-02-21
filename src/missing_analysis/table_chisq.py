"""Check for patterns in missing values via chi-square two-sample tests.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_chisq.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars_to_column
from src.missing_analysis.functions_tables import chisquare_by_column
from src.missing_analysis.functions_tables import create_quantile_dummy
from src.missing_analysis.pretty_index_dict import pretty_index


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1
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
table_chisq_dict = {
    ("Covariates", "chi-squared test statistics"): chisq_cov["chisq"],
    ("Covariates", "p-value"): chisq_cov["pvalue"],
    ("Outcome of interest", "chi-squared test statistics"): chisq_out["chisq"],
    ("Outcome of interest", "p-value"): chisq_out["pvalue"],
}
table_chisq = pd.DataFrame.from_dict(table_chisq_dict).reindex(pretty_index)
table_chisq = table_chisq.rename(pretty_index).dropna(how="all")

# Format DataFrame
idx = pd.IndexSlice
subset_stars = idx[:, idx[:, "p-value"]]
table_chisq = assign_stars_to_column(
    table_chisq, subset_stars, correction=len(table_chisq) - 1
)
table_chisq = table_chisq.fillna(" ")

# Save to latex table
table_chisq.to_latex(
    ppj("OUT_TABLES", "table_chisq.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
