"""Check for missing values mechanism via Welch's paired t-tests.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_missing.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars
from src.missing_analysis.formatting_tables import format_as_percentage
from src.missing_analysis.functions_tables import ttest_by_column
from src.missing_analysis.pretty_index import pretty_index_dict


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
)

# Compute Welch`s t-test p-values and mean comparison
ttest_cov = ttest_by_column(gate_missing, "missing_cov", equal_var=False)
ttest_out = ttest_by_column(gate_missing, "missing_out", equal_var=False)

# Create MultiIndex DataFrame
table_missing = pd.concat(
    [ttest_cov, ttest_out],
    axis=1,
    keys=["Covariates", "Outcome of interest"],
    sort=False,
)

# Format DataFrame
table_missing = table_missing.reindex(pretty_index_dict).rename(pretty_index_dict)
table_missing = table_missing.dropna(how="all")
table_missing = table_missing.rename(
    columns=({"mean1": "Missing", "mean0": "No missing"})
)
idx = pd.IndexSlice
keep_format = [
    "Age",
    "Highest grade achieved",
    "Standardized autonomy index",
    "Standardized risk-tolerance index",
]
rows_to_format = [item for item in table_missing.index if item not in keep_format]
subset = idx[rows_to_format, idx[:, ["Missing", "No missing"]]]
table_missing = format_as_percentage(table_missing, subset)
subset_stars = idx[:, idx[:, "p-value"]]
correction = correction = len(table_missing) - 1
table_missing = assign_stars(table_missing, subset_stars, correction)

# Save to latex table
table_missing.to_latex(
    ppj("OUT_TABLES", "table_missing.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)
