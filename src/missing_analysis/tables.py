"""Check for patterns in missing values and verify randomization integrity at
application and follow-up wave 2.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_integrity.tex``, ``table_missing.tex``, ``table_levene.tex``
and ``table_chisq.tex`` in the "OUT_TABLES" directory.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars_to_column
from src.missing_analysis.formatting_tables import format_as_percentage
from src.missing_analysis.functions_tables import chisquare_by_column
from src.missing_analysis.functions_tables import compute_sample_sizes
from src.missing_analysis.functions_tables import create_quantile_dummy
from src.missing_analysis.functions_tables import levene_by_column
from src.missing_analysis.functions_tables import ttest_by_column


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))


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
# Drop redundant variables from application dataset
gate_app = gate_final.drop(
    ["gateid", "hhincome", "hhincome_w2", "site", "agesqr"], axis=1,
)


# Restrict follow-up wave 2 to individuals who did not attrit
gate_wave2 = gate_app[gate_final["completed_w2"] == 1]


# Compute t-test pvalues and mean comparison
app = ttest_by_column(gate_app, "treatment", equal_var=True)
wave2 = ttest_by_column(gate_wave2, "treatment", equal_var=True)


# Compute sample sizes for treatment and control
app_size = compute_sample_sizes(gate_app)
wave2_size = compute_sample_sizes(gate_wave2)


# Create MultiIndex DataFrame
table_integrity = {
    ("Baseline", "Treatment"): app["mean1"].append(app_size[1]),
    ("Baseline", "Control"): app["mean0"].append(app_size[0]),
    ("Baseline", "p-value"): app["pvalue"],
    ("Follow-up wave 2", "Treatment"): wave2["mean1"].append(wave2_size[1]),
    ("Follow-up wave 2", "Control"): wave2["mean0"].append(wave2_size[0]),
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
columns_to_format = pd.IndexSlice[:, ["Treatment", "Control"]]
table_integrity = format_as_percentage(
    table_integrity, rows_to_format, columns_to_format
)
table_integrity = assign_stars_to_column(
    table_integrity, pd.IndexSlice[:, "p-value"], correction=len(table_integrity) - 1
)


# Save to latex table
table_integrity.to_latex(
    ppj("OUT_TABLES", "table_integrity.tex"),
    float_format="{:.3f}".format,
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
gate_missing["missing_out"] = np.where(np.isnan(gate_final["hhincome_w2"]), 1, 0)


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
keep_format = [
    "Age",
    "Highest grade achieved",
    "Standardized autonomy index",
    "Standardized risk-tolerance index",
]
rows_to_format = [item for item in table_missing.index if item not in keep_format]
columns_to_format = pd.IndexSlice[:, ["Missing", "No missing"]]
table_missing = format_as_percentage(table_missing, rows_to_format, columns_to_format)
table_missing.loc["Missing value(s) in covariates", "Covariates"] = [
    np.nan,
    np.nan,
    np.nan,
]
table_missing.loc["Missing value in outcome", "Outcome of interest"] = [
    np.nan,
    np.nan,
    np.nan,
]
table_missing = assign_stars_to_column(
    table_missing, pd.IndexSlice[:, "p-value"], correction=len(table_missing) - 1
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
table_levene = assign_stars_to_column(
    table_levene, pd.IndexSlice[:, "p-value"], correction=len(table_levene) - 1
)
table_levene = table_levene.fillna(" ")


# Save to latex table
table_levene.to_latex(
    ppj("OUT_TABLES", "table_levene.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)


# CHI-SQUARE TEST
# Create dummy variables from continuous variables
gate_missing["age_p25"] = create_quantile_dummy(gate_missing, "age", 0.25)
gate_missing["age_p25_75"] = create_quantile_dummy(gate_missing, "age", [0.25, 0.75])
gate_missing["age_p75"] = create_quantile_dummy(gate_missing, "age", 0.75, lower=False)

gate_missing["grade_p25"] = create_quantile_dummy(gate_missing, "grade", 0.25)
gate_missing["grade_p25_75"] = create_quantile_dummy(
    gate_missing, "grade", [0.25, 0.75]
)
gate_missing["grade_p75"] = create_quantile_dummy(
    gate_missing, "grade", 0.75, lower=False
)

gate_missing["low_risk_tolerance"] = create_quantile_dummy(
    gate_missing, "risk_tolerance_std", 0.5
)
gate_missing["low_autonomy"] = create_quantile_dummy(gate_missing, "autonomy_std", 0.5)


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
table_chisq = assign_stars_to_column(
    table_chisq, pd.IndexSlice[:, "p-value"], correction=len(table_chisq) - 1
)
table_chisq = table_chisq.fillna(" ")


# Save to latex table
table_chisq.to_latex(
    ppj("OUT_TABLES", "table_chisq.tex"),
    float_format="{:.3g}".format,
    multicolumn_format="c",
)
