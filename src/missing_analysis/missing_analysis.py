"""Check for patterns in missing values and verify randomization integrity at
application and follow-up wave 2.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_integrity.tex``, ``table_missing.tex``, ``table_levene.tex``
and ``table_chisq.tex`` in the "OUT_TABLES" directory.

"""
import numpy as np
import pandas as pd
from formatting import assign_stars
from formatting import format_table
from functions_analysis import chisquare_by_column
from functions_analysis import levene_by_column
from functions_analysis import ttest_by_column

from bld.project_paths import project_paths_join as ppj


# import os
# os.chdir("C:/Projects/badini_garamow/bld/out/data")


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))
# gate_final = pd.read_csv("gate_final.csv")


# Renaming for aesthetic reasons
pretty_index = {
    "treatment": "Treatment",
    "philadelphia": "Philadelphia",
    "pittsburgh": "Pittsburgh",
    "minneapolis": "Minneapolis",
    "duluth": "Duluth",
    "maine": "Maine",
    "age": "Age",
    "age_p25": "35 or younger",
    "age_p25_75": "Between 35 and 50",
    "age_p75": "50 or older",
    "grade": "Highest grade achieved",
    "grade_p25": "High-school or less",
    "grade_p25_75": "Some college education",
    "grade_p75": "Bachelor's degree or more",
    "female": "Female",
    "married": "Married",
    "children": "Has children",
    "not_born_us": "Not born in the US",
    "nonenglish": "Not an English native speaker",
    "white": "White",
    "black": "Black",
    "asian": "Asian",
    "latino": "Latino",
    "other": "Other race",
    "healthproblem": "Has an health problem",
    "employer_healthins": "Health insurance provided by employer",
    "self_employed": "Is self-employed",
    "unemployed": "Unemployed",
    "salaried_worker": "Salaried worker",
    "worked_for_relatives_friends_se": "Worked for self-employed friends",
    "benefits": "Receives unemployment benefits",
    "hhincome_p25": "Yearly income lower than 25k",
    "hhincome_p25_49": "Yearly income between 25k and 49k",
    "hhincome_p50_74": "Yearly income between 50k and 74k",
    "hhincome_p75_99": "Yearly income between 75k and 99k",
    "hhincome_p100": "Yearly income higher than 100k",
    "badcredit": "Has bad credit history",
    "autonomy_std": "Standardized autonomy index",
    "risk_tolerance_std": "Standardized risk-tolerance index",
    "sample_size": "Sample size",
    "completed_w2": "Percent of baseline sample",
    "missing_cov": "Missing value(s) in covariates",
    "missing_out": "Missing value in outcome",
}


# RANDOMIZATION INTEGRITY CHECK


# Drop redundant variables
gate_app = gate_final.drop(
    ["gateid", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1,
)


# Restrict follow-up wave 2 to individuals who did not attrit
gate_wave2 = gate_app[gate_final["completed_w2"] == 1]


# Compute t-test pvalues and mean comparison
app = ttest_by_column(gate_app, "treatment", equal_var=True)
wave2 = ttest_by_column(gate_wave2, "treatment", equal_var=True)


# Include sample sizes
sample_sizes = []
for df in [gate_app, gate_wave2]:
    for dummy in (1, 0):
        sample_size = pd.Series(
            df.groupby("treatment").size()[dummy], index=["sample_size"]
        )
        sample_sizes.append(sample_size)


# Create MultiIndex DataFrame
table_integrity = {
    ("Baseline", "Treatment group"): app["mean1"].append(sample_sizes[0]),
    ("Baseline", "Control group"): app["mean0"].append(sample_sizes[1]),
    ("Baseline", "p-value"): app["pvalue"],
    ("Follow-up wave 2", "Treatment group"): wave2["mean1"].append(sample_sizes[2]),
    ("Follow-up wave 2", "Control group"): wave2["mean0"].append(sample_sizes[3]),
    ("Follow-up wave 2", "p-value"): wave2["pvalue"],
}
table_integrity = pd.DataFrame.from_dict(table_integrity).reindex(pretty_index)
table_integrity = table_integrity.rename(pretty_index).dropna(how="all")


# Format DataFrame
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
columns_to_format = ["Treatment group", "Control group"]
table_integrity = format_table(
    table_integrity, rows_to_format, columns_to_format, "{:,.2%}"
)
table_integrity = format_table(
    table_integrity, "Sample size", columns_to_format, "{:,.0f}"
)
table_integrity = assign_stars(
    table_integrity, "p-value", correction=len(table_integrity) - 1
)


# Save to latex table
table_integrity.to_latex(
    ppj("OUT_TABLES", "table_integrity.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)


# MISSING VALUES ANALYSIS


# Drop redundant variables
gate_missing = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1
)


# Dummy variables to indicate whether missing value(s) are present in covariates...
gate_missing["missing_cov"] = np.where(np.isnan(gate_missing).any(axis=1), 1, 0)
# ... or in outcome
gate_missing["missing_out"] = np.where(np.isnan(gate_final["hhincome_w2"]), 1, 0)


# Compute Welch`s t-test p-values and mean comparison
ttest_cov = ttest_by_column(gate_missing, "missing_cov", equal_var=False)
ttest_out = ttest_by_column(gate_missing, "missing_out", equal_var=False)


# Create MultiIndex DataFrame
table_missing_dict = {
    ("Covariates", "Missing value(s)"): ttest_cov["mean1"],
    ("Covariates", "No missing value(s)"): ttest_cov["mean0"],
    ("Covariates", "p-value"): ttest_cov["pvalue"],
    ("Outcome of interest", "Missing value"): ttest_out["mean1"],
    ("Outcome of interest", "No missing value"): ttest_out["mean0"],
    ("Outcome of interest", "p-value"): ttest_out["pvalue"],
}
table_missing = pd.DataFrame.from_dict(table_missing_dict).reindex(pretty_index)
table_missing = table_missing.rename(pretty_index).dropna(how="all")


# Format DataFrame
keep_format = [
    "Age",
    "Highest grade achieved",
    "Standardized autonomy index",
    "Standardized risk-tolerance index",
    "Missing value(s) in covariates",
    "Missing value in outcome",
]
rows_to_format = [item for item in table_missing.index if item not in keep_format]
columns_to_format = [
    "Missing value(s)",
    "No missing value(s)",
    "Missing value",
    "No missing value",
]
table_missing = format_table(
    table_missing, rows_to_format, columns_to_format, "{:,.2%}"
)
table_missing = format_table(
    table_missing,
    "Missing value(s) in covariates",
    ["Missing value", "No missing value"],
    "{:,.2%}",
)
table_missing = format_table(
    table_missing,
    "Missing value in outcome",
    ["Missing value(s)", "No missing value(s)"],
    "{:,.2%}",
)
table_missing = assign_stars(
    table_missing, "p-value", correction=len(table_missing) - 1
)


# Save to latex table
table_missing.to_latex(
    ppj("OUT_TABLES", "table_missing.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)


# CONSISTENCY CHECK: LEVENE TEST FOR EQUAL VARIANCES


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
table_levene = assign_stars(table_levene, "p-value", correction=len(table_levene) - 1)
table_levene = table_levene.fillna(" ")


# Save to latex table
table_levene.to_latex(
    ppj("OUT_TABLES", "table_levene.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)


# CHI-SQUARE TEST


# Create dummy variables from continuous variables
gate_missing["age_p25"] = np.where(
    np.isnan(gate_missing["age"]),
    np.nan,
    np.where(gate_missing["age"] <= gate_missing["age"].quantile(0.25), 1, 0),
)
gate_missing["age_p25_75"] = np.where(
    np.isnan(gate_missing["age"]),
    np.nan,
    np.where(
        (gate_missing["age"] > gate_missing["age"].quantile(0.25))
        & (gate_missing["age"] < gate_missing["age"].quantile(0.75)),
        1,
        0,
    ),
)
gate_missing["age_p75"] = np.where(
    np.isnan(gate_missing["age"]),
    np.nan,
    np.where(gate_missing["age"] >= gate_missing["age"].quantile(0.75), 1, 0),
)


gate_missing["grade_p25"] = np.where(
    np.isnan(gate_missing["grade"]),
    np.nan,
    np.where(gate_missing["grade"] <= gate_missing["grade"].quantile(0.25), 1, 0),
)
gate_missing["grade_p25_75"] = np.where(
    np.isnan(gate_missing["grade"]),
    np.nan,
    np.where(
        (gate_missing["grade"] > gate_missing["grade"].quantile(0.25))
        & (gate_missing["grade"] < gate_missing["grade"].quantile(0.75)),
        1,
        0,
    ),
)
gate_missing["grade_p75"] = np.where(
    np.isnan(gate_missing["grade"]),
    np.nan,
    np.where(gate_missing["grade"] >= gate_missing["grade"].quantile(0.75), 1, 0),
)


gate_missing["low_risk_tolerance"] = np.where(
    np.isnan(gate_missing["risk_tolerance_std"]),
    np.nan,
    np.where(
        gate_missing["risk_tolerance_std"]
        <= gate_missing["risk_tolerance_std"].median(),
        1,
        0,
    ),
)
gate_missing["low_autonomy"] = np.where(
    np.isnan(gate_missing["autonomy_std"]),
    np.nan,
    np.where(
        gate_missing["autonomy_std"] <= gate_missing["autonomy_std"].median(), 1, 0
    ),
)


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
table_chisq = assign_stars(table_chisq, "p-value", correction=len(table_chisq) - 1)
table_chisq = table_chisq.fillna(" ")


# Save to latex table
table_chisq.to_latex(
    ppj("OUT_TABLES", "table_chisq.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
