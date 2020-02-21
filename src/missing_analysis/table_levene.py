"""Perform paired Levene's test for equal variances as consistency check.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_levene.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars_to_column
from src.missing_analysis.functions_tables import levene_by_column
from src.missing_analysis.pretty_index_dict import pretty_index


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1
)

# Compute Levene statistic and p-values
levene_cov = levene_by_column(gate_missing, "missing_cov")
levene_out = levene_by_column(gate_missing, "missing_out")

# Create MultiIndex DataFrame
table_levene_dict = {
    ("Covariates", "Levene's test statistic"): levene_cov["levene_stat"],
    ("Covariates", "p-value"): levene_cov["pvalue"],
    ("Outcome of interest", "Levene's test statistic"): levene_out["levene_stat"],
    ("Outcome of interest", "p-value"): levene_out["pvalue"],
}
table_levene = pd.DataFrame.from_dict(table_levene_dict).reindex(pretty_index)
table_levene = table_levene.rename(pretty_index).dropna(how="all")

# Format DataFrame
idx = pd.IndexSlice
subset_stars = idx[:, idx[:, "p-value"]]
table_levene = assign_stars_to_column(
    table_levene, subset_stars, correction=len(table_levene) - 1
)
table_levene = table_levene.fillna(" ")

# Save to latex table
table_levene.to_latex(
    ppj("OUT_TABLES", "table_levene.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
