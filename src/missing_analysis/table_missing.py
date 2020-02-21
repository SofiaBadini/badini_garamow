"""Check for patterns in missing values via Welch's paired t-tests.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_missing.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars_to_column
from src.missing_analysis.formatting_tables import format_as_percentage
from src.missing_analysis.functions_tables import ttest_by_column
from src.missing_analysis.pretty_index_dict import pretty_index

# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1
)

# Compute Welch`s t-test p-values and mean comparison
ttest_cov = ttest_by_column(gate_missing, "missing_cov", equal_var=False)
ttest_out = ttest_by_column(gate_missing, "missing_out", equal_var=False)

# Create MultiIndex DataFrame
table_missing_dict = {
    ("Covariates", "Missing"): ttest_cov["mean1"],
    ("Covariates", "No missing"): ttest_cov["mean0"],
    ("Covariates", "p-value"): ttest_cov["pvalue"],
    ("Outcome of interest", "Missing"): ttest_out["mean1"],
    ("Outcome of interest", "No missing"): ttest_out["mean0"],
    ("Outcome of interest", "p-value"): ttest_out["pvalue"],
}
table_missing = pd.DataFrame.from_dict(table_missing_dict).reindex(pretty_index)
table_missing = table_missing.rename(pretty_index).dropna(how="all")

# Format DataFrame
idx = pd.IndexSlice
keep_format = [
    "Age",
    "Highest grade achieved",
    "Standardized autonomy index",
    "Standardized risk-tolerance index",
]
rows_to_format = [item for item in table_missing.index if item not in keep_format]
table_missing = format_as_percentage(
    table_missing, idx[rows_to_format, idx[:, ["Missing", "No missing"]]]
)
subset_stars = idx[:, idx[:, "p-value"]]
table_missing = assign_stars_to_column(
    table_missing, subset_stars, correction=len(table_missing) - 1
)

# Save to latex table
table_missing.to_latex(
    ppj("OUT_TABLES", "table_missing.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)
