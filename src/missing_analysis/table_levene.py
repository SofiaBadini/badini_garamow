"""Perform paired Levene's test for equal variances as consistency check.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_levene.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars
from src.missing_analysis.functions_tables import levene_by_column
from src.missing_analysis.pretty_index import pretty_index_dict


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
)

# Compute Levene statistic and p-values
levene_cov = levene_by_column(gate_missing, "missing_cov")
levene_out = levene_by_column(gate_missing, "missing_out")

# Create MultiIndex dataframe
table_levene = pd.concat(
    [levene_cov, levene_out],
    axis=1,
    keys=["Covariates", "Outcome of interest"],
    sort=False,
)

# Format DataFrame
table_levene = table_levene.reindex(pretty_index_dict).rename(pretty_index_dict)
table_levene = table_levene.dropna(how="all")
idx = pd.IndexSlice
subset = idx[:, idx[:, "p-value"]]
correction = len(table_levene) - 1
table_levene = assign_stars(table_levene, subset, correction)
table_levene = table_levene.fillna(" ")

# Save to latex table
table_levene.to_latex(
    ppj("OUT_TABLES", "table_levene.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
