"""Graphical analysis of missing data. Produces heatmap to show nullity
correlation and nullity matrix to visualize pattern in missing data.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
plots to ``heatmap_nan.png`` and ``matrix_nan.png`` in the "OUT_FIGURES" directory.

"""
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


# import os
# os.chdir("C:/Projects/badini_garamow/bld/out/data")


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))
# gate_final = pd.read_csv("gate_final.csv")


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
    "HAS AN HEALTH PROBLEM",
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
        "agesqr",
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


# Sort dataset according to nan, by index and by column
index_missing = gate_for_plot.isna().sum().sort_values().index
sorted_by_missing = msno.nullity_sort(gate_for_plot[index_missing])


# Sort dataset by characteristic category
index_category = pd.Index(new_names)
sorted_by_category = gate_for_plot[index_category]


# To avoid cropping pictures when saving them
plt.tight_layout()


# Matrix
matrix_nan = msno.matrix(sorted_by_missing)
matrix_nan.set_ylabel("INDEX OF OBSERVATIONS", labelpad=0, fontsize=18)
# Outcome in bold
matrix_nan.get_xticklabels()[19].set_fontweight("bold")


# Heatmap
heatmap_nan = msno.heatmap(sorted_by_category, vmin=0, cmap="OrRd")
# Outcome in bold
heatmap_nan.get_xticklabels()[16].set_fontweight("bold")
heatmap_nan.get_yticklabels()[16].set_fontweight("bold")
# Interesting fact:
# When plotting heatmaps with seaborn (on which the "missingno" library builds)
# the first and the last row is cut in halve, because of a bug in the matplotlib
# regression between 3.1.0 and 3.1.1
# We are correcting it this way:
bottom, top = heatmap_nan.get_ylim()
heatmap_nan.set_ylim(bottom + 0.5, top - 0.5)
positions = np.array([1, 3, 5, 8, 10, 14, 16])
labels = ["BACKGROUND", "HOUSEHOLD", "FINANCE", "HEALTH", "EMPLOYMENT", "PERSONALITY"]
heatmap_nan.hlines(positions, xmin=0, xmax=positions, lw=8, color="white")
for position, label in zip(positions, labels):
    heatmap_nan.text(position + 0.35, position + 0.35, label, fontsize=14)


# Save plots as png
matrix_nan.figure.savefig(ppj("OUT_FIGURES", "matrix_nan.png"), bbox_inches="tight")
heatmap_nan.figure.savefig(ppj("OUT_FIGURES", "heatmap_nan.png"), bbox_inches="tight")

# matrix_nan.figure.savefig("matrix_nan.png", bbox_inches="tight")
# heatmap_nan.figure.savefig("heatmap_nan.png", bbox_inches="tight")
