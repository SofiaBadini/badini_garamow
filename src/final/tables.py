"""Format results of missing data mechanism analysis, stored as .csv files in the
"OUT_ANALYSIS" directory, as latex tables, and sore them in the "OUT_TABLES" directory.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.auxiliary.formatting import assign_stars
from src.auxiliary.formatting import format_as_percentage
from src.auxiliary.pretty_index import pretty_index_dict


def chisq_df_as_table():
    """Format ``chisq_df.csv`` and save the result to ``table_chisq.tex``.

    """
    chisq_df = pd.read_csv(
        ppj("OUT_ANALYSIS", "chisq_df.csv"), index_col=[0], header=[0, 1]
    )
    chisq_df = chisq_df.reindex(pretty_index_dict).rename(pretty_index_dict)
    chisq_df = chisq_df.dropna(how="all")
    idx = pd.IndexSlice
    subset = idx[:, idx[:, "p-value"]]
    correction = len(chisq_df) - 1
    chisq_df = assign_stars(chisq_df, subset, correction)
    chisq_df = chisq_df.fillna(" ")
    chisq_df.to_latex(
        ppj("OUT_TABLES", "table_chisq.tex"),
        float_format="{:.3g}".format,
        multicolumn_format="c",
    )


def integrity_df_as_table():
    """Format ``integrity_df.csv`` and save the result to ``table_integrity.tex``.

    """
    integrity_df = pd.read_csv(
        ppj("OUT_ANALYSIS", "integrity_df.csv"), index_col=[0], header=[0, 1]
    )
    integrity_df = integrity_df.reindex(pretty_index_dict)
    integrity_df = integrity_df.rename(pretty_index_dict).dropna(how="all")
    new_values = integrity_df.loc["Percent of baseline sample", "Baseline"].values
    integrity_df.loc["Percent of baseline sample", "Follow-up wave 2"] = new_values
    integrity_df.loc["Percent of baseline sample", "Baseline"] = [1, 1, np.nan]
    keep_format = [
        "Age",
        "Highest grade achieved",
        "Sample size",
        "Standardized autonomy index",
        "Standardized risk-tolerance index",
    ]
    rows_to_format = [item for item in integrity_df.index if item not in keep_format]
    idx = pd.IndexSlice
    subset = idx[rows_to_format, idx[:, ["Treatment", "Control"]]]
    integrity_df = format_as_percentage(integrity_df, subset)
    subset_stars = idx[:, idx[:, "p-value"]]
    correction = len(integrity_df) - 1
    integrity_df = assign_stars(integrity_df, subset_stars, correction)
    integrity_df.to_latex(
        ppj("OUT_TABLES", "table_integrity.tex"),
        float_format="{:.3f}".format,
        na_rep=" ",
        multicolumn_format="c",
    )


def levene_df_as_table():
    """Format ``levene_df.csv`` and save the result to ``levene_table.tex``.

    """
    levene_df = pd.read_csv(
        ppj("OUT_ANALYSIS", "levene_df.csv"), index_col=[0], header=[0, 1]
    )
    levene_df = levene_df.reindex(pretty_index_dict).rename(pretty_index_dict)
    levene_df = levene_df.dropna(how="all")
    idx = pd.IndexSlice
    subset = idx[:, idx[:, "p-value"]]
    correction = len(levene_df) - 1
    levene_df = assign_stars(levene_df, subset, correction)
    levene_df = levene_df.fillna(" ")
    levene_df.to_latex(
        ppj("OUT_TABLES", "table_levene.tex"),
        float_format="{:.3g}".format,
        multicolumn_format="c",
    )


def logistic_df_as_table():
    """Format ``logistic_df.csv`` and save the result to ``logistic_table.csv``.

    """
    logistic_df = pd.read_csv(
        ppj("OUT_ANALYSIS", "logistic_df.csv"), index_col=[0], header=[0, 1]
    )
    logistic_df = logistic_df.reindex(pretty_index_dict).rename(pretty_index_dict)
    logistic_df = logistic_df.dropna(how="all")
    idx = pd.IndexSlice
    subset = idx[:, idx[:, "p-value"]]
    logistic_df = assign_stars(logistic_df, subset)
    logistic_df = logistic_df.fillna(" ")
    logistic_df.to_latex(
        ppj("OUT_TABLES", "table_logistic.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )


def welch_df_as_table():
    """Format ``welch_df.csv`` and save the result to ``table_welch.tex``.

    """
    welch_df = pd.read_csv(
        ppj("OUT_ANALYSIS", "welch_df.csv"), index_col=[0], header=[0, 1]
    )
    welch_df = welch_df.reindex(pretty_index_dict).rename(pretty_index_dict)
    welch_df = welch_df.dropna(how="all")
    welch_df = welch_df.rename(columns=({"mean1": "Missing", "mean0": "No missing"}))
    idx = pd.IndexSlice
    keep_format = [
        "Age",
        "Highest grade achieved",
        "Standardized autonomy index",
        "Standardized risk-tolerance index",
    ]
    rows_to_format = [item for item in welch_df.index if item not in keep_format]
    subset = idx[rows_to_format, idx[:, ["Missing", "No missing"]]]
    welch_df = format_as_percentage(welch_df, subset)
    subset_stars = idx[:, idx[:, "p-value"]]
    correction = len(welch_df) - 1
    welch_df = assign_stars(welch_df, subset_stars, correction)
    welch_df.to_latex(
        ppj("OUT_TABLES", "table_welch.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )


def complete_controls_as_table():
    """Format ``complete_controls_coeff.csv`` and ``complete_controls_summary.csv``
    and save the result to ``table_complete_controls_coeff.tex`` and
    ``table_complete_controls_summary.tex``.

    """
    complete_controls_coeff = pd.read_csv(
        ppj("OUT_ANALYSIS", "complete_controls_coeff_df.csv"), index_col=0
    )
    complete_controls_summary = pd.read_csv(
        ppj("OUT_ANALYSIS", "complete_controls_summary_df.csv"), index_col=0
    )
    complete_controls_coeff = complete_controls_coeff.reindex(pretty_index_dict).rename(
        pretty_index_dict
    )
    complete_controls_coeff = complete_controls_coeff.dropna(how="all")
    subset = pd.IndexSlice[:, "p-value"]
    complete_controls_coeff = assign_stars(complete_controls_coeff, subset)
    complete_controls_coeff.to_latex(
        ppj("OUT_TABLES", "table_complete_controls_coeff.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )
    complete_controls_summary.T.to_latex(
        ppj("OUT_TABLES", "table_complete_controls_summary.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )


def complete_no_controls_as_table():
    """Format ``complete_no_controls_coeff.csv`` and ``complete_no_controls_summary.csv``
    and save the result to ``table_complete_no_controls_coeff.tex`` and
    ``complete_no_controls_summary.tex``.

    """
    complete_no_controls_coeff = pd.read_csv(
        ppj("OUT_ANALYSIS", "complete_no_controls_coeff_df.csv"), index_col=0
    )
    complete_no_controls_summary = pd.read_csv(
        ppj("OUT_ANALYSIS", "complete_no_controls_summary_df.csv"), index_col=0
    )
    complete_no_controls_coeff = complete_no_controls_coeff.reindex(
        pretty_index_dict
    ).rename(pretty_index_dict)
    complete_no_controls_coeff = complete_no_controls_coeff.dropna(how="all")
    complete_no_controls_coeff.to_latex(
        ppj("OUT_TABLES", "table_complete_no_controls_coeff.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )
    complete_no_controls_summary.T.to_latex(
        ppj("OUT_TABLES", "table_complete_no_controls_summary.tex"),
        float_format="{:.3g}".format,
        na_rep=" ",
        multicolumn_format="c",
    )


if __name__ == "__main__":
    chisq_df_as_table()
    integrity_df_as_table()
    levene_df_as_table()
    logistic_df_as_table()
    welch_df_as_table()
    complete_controls_as_table()
    complete_no_controls_as_table()
