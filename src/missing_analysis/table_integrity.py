"""Verify randomization integrity at application and follow-up wave 2.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
table to ``table_integrity.tex`` in the "OUT_TABLES" directory.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars
from src.missing_analysis.formatting_tables import format_as_percentage
from src.missing_analysis.functions_tables import compute_sample_sizes
from src.missing_analysis.functions_tables import ttest_by_column
from src.missing_analysis.pretty_index import pretty_index_dict

# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables from application dataset
gate_app = gate_final.drop(
    ["gateid", "hhincome", "hhincome_w2", "site", "missing_cov", "missing_out"], axis=1,
)

# Restrict follow-up wave 2 to individuals who did not attrit
gate_wave2 = gate_app[gate_final["missing_out"] == 0]

# Compute t-test pvalues and mean comparison
app = ttest_by_column(gate_app, "treatment", equal_var=True)
wave2 = ttest_by_column(gate_wave2, "treatment", equal_var=True)

# Compute sample sizes for treatment and control
app_size = compute_sample_sizes(gate_app, "treatment")
wave2_size = compute_sample_sizes(gate_wave2, "treatment")

# Create MultiIndex DataFrame
table_integrity = {
    ("Baseline", "Treatment"): app["mean1"].append(app_size[1]),
    ("Baseline", "Control"): app["mean0"].append(app_size[0]),
    ("Baseline", "p-value"): app["p-value"],
    ("Follow-up wave 2", "Treatment"): wave2["mean1"].append(wave2_size[1]),
    ("Follow-up wave 2", "Control"): wave2["mean0"].append(wave2_size[0]),
    ("Follow-up wave 2", "p-value"): wave2["p-value"],
}
table_integrity = pd.DataFrame.from_dict(table_integrity).reindex(pretty_index_dict)

# Format DataFrame
table_integrity = table_integrity.rename(pretty_index_dict).dropna(how="all")
new_values = table_integrity.loc["Percent of baseline sample", "Baseline"].values
table_integrity.loc["Percent of baseline sample", "Follow-up wave 2"] = new_values
table_integrity.loc["Percent of baseline sample", "Baseline"] = [1, 1, np.nan]
keep_format = [
    "Age",
    "Highest grade achieved",
    "Sample size",
    "Standardized autonomy index",
    "Standardized risk-tolerance index",
]
rows_to_format = [item for item in table_integrity.index if item not in keep_format]
idx = pd.IndexSlice
subset = idx[rows_to_format, idx[:, ["Treatment", "Control"]]]
table_integrity = format_as_percentage(table_integrity, subset)
subset_stars = idx[:, idx[:, "p-value"]]
correction = len(table_integrity) - 1
table_integrity = assign_stars(table_integrity, subset_stars, correction)

# Save to latex table
table_integrity.to_latex(
    ppj("OUT_TABLES", "table_integrity.tex"),
    float_format="{:.3f}".format,
    na_rep=" ",
    multicolumn_format="c",
)
