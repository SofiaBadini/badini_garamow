"""Visualization of the missing data pattern in ``gate_final.csv``,
stored in the "OUT_DATA" directory.

"""
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

old_labels = [
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
new_labels = [
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
mapping_aes = dict(zip(old_labels, new_labels))

# Drop redundant variables from dataset
gate_plot = gate_final.drop(
    [
        "treatment",
        "completed_w2",
        "gateid",
        "site",
        "hhincome_25k",
        "hhincome_25_49k",
        "hhincome_50_74k",
        "hhincome_75_99k",
        "hhincome_100k",
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
gate_plot["race"] = np.where(
    (np.isnan(gate_final["white"]))
    & (np.isnan(gate_final["black"]))
    & (np.isnan(gate_final["latino"]))
    & (np.isnan(gate_final["asian"]))
    & (np.isnan(gate_final["other"])),
    np.nan,
    1,
)

gate_plot = gate_plot.rename(columns=mapping_aes)


plt.tight_layout()


def create_matrix_nan():
    """Create nullity matrix and save the plot to ``matrix_nan.png``
    in the "OUT_DATA" directory.

    """
    index_missing = gate_plot.isna().sum().sort_values().index
    sorted_by_missing = msno.nullity_sort(gate_plot[index_missing])
    matrix_nan = msno.matrix(sorted_by_missing)
    matrix_nan.set_ylabel("INDEX OF OBSERVATIONS", labelpad=0, fontsize=18)
    matrix_nan.get_xticklabels()[19].set_fontweight("bold")
    matrix_nan.figure.savefig(ppj("OUT_FIGURES", "matrix_nan.png"), bbox_inches="tight")


create_matrix_nan()


def create_heatmap_nan():
    """Create nullity correlation heatmap and save the plot to ``matrix_nan.png``
    in the "OUT_DATA" directory.

    """
    index_category = pd.Index(new_labels)
    sorted_by_category = gate_plot[index_category]
    heatmap_nan = msno.heatmap(sorted_by_category, vmin=0, cmap="OrRd")
    heatmap_nan.get_xticklabels()[16].set_fontweight("bold")
    heatmap_nan.get_yticklabels()[16].set_fontweight("bold")
    # Interesting fact:
    # When plotting heatmaps with seaborn (on which the "missingno" library
    # builds), the first and the last row is cut in halve, because of a bug
    # in the matplotlib regression between 3.1.0 and 3.1.1
    # We are correcting it this way:
    bottom, top = heatmap_nan.get_ylim()
    heatmap_nan.set_ylim(bottom + 0.5, top - 0.5)
    positions = np.array([1, 3, 5, 8, 10, 14, 16])
    labels = [
        "BACKGROUND",
        "HOUSEHOLD",
        "FINANCE",
        "HEALTH",
        "EMPLOYMENT",
        "PERSONALITY",
    ]
    heatmap_nan.hlines(positions, xmin=0, xmax=positions, lw=8, color="white")
    for position, label in zip(positions, labels):
        heatmap_nan.text(position + 0.35, position + 0.35, label, fontsize=14)
    heatmap_nan.figure.savefig(
        ppj("OUT_FIGURES", "heatmap_nan.png"), bbox_inches="tight"
    )


create_heatmap_nan()
