"""Produce nullity correlation heatmap and nullity matrix.

Visualize missing data pattern in ``gate_final.csv``, stored in the "OUT_DATA"
directory, and save plots to ``heatmap_nan.png`` and ``matrix_nan.png`` in the
"OUT_FIGURES" directory.

"""
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.functions_plots import heatmap_nan
from src.missing_analysis.functions_plots import matrix_nan


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Mapping names for aesthetic reasons
old_names = [
    "age",
    "female",
    "grade",
    "nonenglish",
    "not_born_us",
    "race",
    "children",
    "married",
    "badcredit",
    "benefits",
    "hhincome",
    "employer_healthins",
    "healthproblem",
    "salaried_worker",
    "self_employed",
    "unemployed",
    "worked_for_relatives_friends_se",
    "autonomy_std",
    "risk_tolerance_std",
    "hhincome_w2",
]
new_names = [
    "AGE",
    "FEMALE",
    "HIGHEST GRADE ACHIEVED",
    "NOT AN ENGLISH NATIVE \n SPEAKER",
    "NOT BORN IN THE US",
    "RACE",
    "HAS CHILDREN",
    "MARRIED",
    "HAS BAD CREDIT HISTORY",
    "RECEIVES UNEMPLOYMENT BENEFITS",
    "HOUSEHOLD INCOME",
    "HEALTH INSURANCE PROVIDED \n BY EMPLOYER",
    "HAS A HEALTH PROBLEM",
    "SALARIED WORKER",
    "SELF-EMPLOYED",
    "UNEMPLOYED",
    "WORKED FOR SELF-EMPLOYED \n FRIENDS",
    "AUTONOMY INDEX",
    "RISK-TOLERANCE INDEX",
    "HOUSEHOLD INCOME AT \n WAVE 2",
]
mapping_aes = dict(zip(old_names, new_names))

# Drop redundant variables from dataset
gate_for_plot = gate_final.drop(
    [
        "treatment",
        "completed_w2",
        "gateid",
        "site",
        "hhincome_p25",
        "hhincome_p25_49",
        "hhincome_p50_74",
        "hhincome_p75_99",
        "hhincome_p100",
        "duluth",
        "minneapolis",
        "maine",
        "philadelphia",
        "pittsburgh",
        "white",
        "black",
        "latino",
        "asian",
        "other",
        "missing_cov",
        "missing_out",
    ],
    axis=1,
)

# Create new variable "race"
gate_for_plot["race"] = np.where(
    (np.isnan(gate_final["white"]))
    & (np.isnan(gate_final["black"]))
    & (np.isnan(gate_final["latino"]))
    & (np.isnan(gate_final["asian"]))
    & (np.isnan(gate_final["other"])),
    np.nan,
    1,
)

# Rename columns
gate_for_plot = gate_for_plot.rename(columns=mapping_aes)

# To avoid cropping pictures when saving them
plt.tight_layout()

# Sort dataset according to nan, by index and by column
index_missing = gate_for_plot.isna().sum().sort_values().index
sorted_by_missing = msno.nullity_sort(gate_for_plot[index_missing])

# Matrix
matrix_nan = matrix_nan(sorted_by_missing)

# Sort dataset by characteristic category
index_category = pd.Index(new_names)
sorted_by_category = gate_for_plot[index_category]

# Heatmap
heatmap_nan = heatmap_nan(sorted_by_category)

# Save plots as png
matrix_nan.figure.savefig(ppj("OUT_FIGURES", "matrix_nan.png"), bbox_inches="tight")
heatmap_nan.figure.savefig(ppj("OUT_FIGURES", "heatmap_nan.png"), bbox_inches="tight")
