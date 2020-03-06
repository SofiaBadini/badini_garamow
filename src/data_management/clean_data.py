"""Pre-process data from the original GATE datasets.

Process original data stored in ``application.csv`` and ``wave2.csv`` located
in the "IN_DATA" directory, and save the final datasets to ``gate_long.csv``
and ``gate_final.csv`` in the "OUT_DATA" directory.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj


# Load datasets
filenames = [
    "info_original_variables.csv",
    "application.csv",
    "wave2.csv",
    "final_variables.csv",
]
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(ppj("IN_DATA", filename)))


# Create dataset of variables of interest
variables_name = dataframes[0]["Variable"].to_numpy()
variables_baseline = dataframes[1][dataframes[1].columns.intersection(variables_name)]
variables_w2 = dataframes[2][dataframes[2].columns.intersection(variables_name)]
variables_w2 = variables_w2.drop(["treatment"], axis=1)
gate_df = pd.merge(variables_baseline, variables_w2, on="gateid", how="left")


# Rename variables
mapping = {
    "race_black_not_hispanic": "black",
    "race_white_not_hispanic": "white",
    "race_asian": "asian",
    "has_children_in_hh": "children",
    "disability": "healthproblem",
    "currently_receiving_ui_benefits": "benefits",
    "working_in_wage_and_salary_job": "salaried_worker",
    "self_employed_at_application": "self_employed",
    "household_income": "hhincome",
    "w2_completed": "completed_w2",
}
gate_df.rename(columns=mapping, inplace=True)


# Convert appropriate values to NaN.
# -4 and -2 are coded as "Refused" and "Don't know" respectively.
# -1 indicates a legitimate skip, so it is case specific.
gate_df = gate_df.replace([-4, -2], np.nan)


# Replace -1 with 0, as respondents claimed to work a salary job
# or to have no self-employed relatives or friends in previous
# answers.
gate_df["self_employed"] = gate_df["self_employed"].replace(-1, 0)
gate_df["worked_for_relatives_friends_se"] = gate_df[
    "worked_for_relatives_friends_se"
].replace(-1, 0)


# Create dummy variables from categorical ones
dummies = [0, 1]

mapping_site = {
    1: "philadelphia",
    2: "pittsburgh",
    3: "minneapolis",
    4: "duluth",
    5: "maine",
}
for key in mapping_site.keys():
    gate_df[mapping_site.get(key)] = np.where(
        np.isnan(gate_df["site"]), np.nan, np.where(gate_df["site"] == key, 1, 0)
    )

conditions = [(gate_df["language"] == 1), (gate_df["language"] >= 2)]
gate_df["nonenglish"] = pd.Series(np.select(conditions, dummies, default=np.nan))

conditions = [
    ((gate_df["marital_status"] == 1) | (gate_df["marital_status"] == 2)),
    (gate_df["marital_status"] >= 3),
]
gate_df["married"] = pd.Series(np.select(conditions, dummies, default=np.nan))

conditions = [
    ((gate_df["has_health_insurance"] == 0) | (gate_df["health_insurance_source"] > 1)),
    (gate_df["health_insurance_source"] == 1),
]
gate_df["employer_healthins"] = pd.Series(
    np.select(conditions, dummies, default=np.nan)
)

# Recover yearly income threshold from category indicated
gate_df["hhincome_25k"] = np.where(
    (
        (gate_df["hhincome"] == 0)
        | (gate_df["hhincome"] == 1)
        | (gate_df["hhincome"] == 2)
        | (gate_df["hhincome"] == 3)
    ),
    1,
    0,
)
gate_df["hhincome_25_49k"] = np.where(
    ((gate_df["hhincome"] == 4) | (gate_df["hhincome"] == 5)), 1, 0
)
gate_df["hhincome_50_74k"] = np.where((gate_df["hhincome"] == 6), 1, 0)
gate_df["hhincome_75_99k"] = np.where((gate_df["hhincome"] == 7), 1, 0)
gate_df["hhincome_100k"] = np.where((gate_df["hhincome"] == 8), 1, 0)
# Observation with no income indicated must be converted to NaN.
index = gate_df[gate_df["hhincome"].isnull()].index.tolist()
filter_col = [col for col in gate_df if col.startswith("hhincome_p")]
gate_df.loc[index, filter_col] = np.nan


# Create new variables from existing ones
conditions = [
    ((gate_df["self_employed"] == 1) | (gate_df["salaried_worker"] == 1)),
    ((gate_df["self_employed"] == 0) & (gate_df["salaried_worker"] == 0)),
]
gate_df["unemployed"] = pd.Series(np.select(conditions, dummies, default=np.nan))

gate_df["female"] = np.where(
    np.isnan(gate_df["gender"]), np.nan, np.where(gate_df["gender"] == 0, 1, 0)
)

gate_df["not_born_us"] = np.where(
    np.isnan(gate_df["born_us"]), np.nan, np.where(gate_df["born_us"] == 1, 0, 1)
)

conditions = [
    (
        (gate_df["has_credit_history"] == 0)
        | (gate_df["has_credit_history_problem"] == 0)
    ),
    (gate_df["has_credit_history_problem"] == 1),
]
gate_df["badcredit"] = pd.Series(np.select(conditions, dummies, default=np.nan))

# Aggregate racial categories
gate_df["latino"] = np.where(
    (gate_df["race_white_hispanic"] == 1) | (gate_df["race_black_hispanic"] == 1), 1, 0,
)
gate_df["other"] = np.where(
    (gate_df["race_american_indian_alaskan"] == 1)
    | (gate_df["race_hawaiian_pacific_islander"] == 1)
    | (gate_df["race_other"] == 1),
    1,
    0,
)
# Observation with no race indicated must be converted to NaN.
race_df = gate_df[(gate_df.columns[pd.Series(gate_df.columns).str.startswith("race")])]
index = race_df[race_df.isnull().any(axis=1)].index
gate_df.loc[index, ["latino", "other"]] = np.nan


# Information on household income at wave2, our outcome of interest.
# Respondents were asked to indicate their household income by
# writing a number or selecting an income category ("einc_c3" or "einc_c2")
# When a number is missing, Fairlie et al. impute the mean of the
# income category indicated.
conditions = [(gate_df["einc"] >= 0)]
for income in ("einc_c3", "einc_c2"):
    for value in list(range(1, 7)):
        condition = ((gate_df["einc"] == -1) | (np.isnan(gate_df["einc"]))) & (
            gate_df[income] == value
        )
        conditions.append(condition)

einc_c3_cat = list(range(2500, 32500, 5000))
einc_c2_cat = list(range(37500, 127500, 15000))
choices = [gate_df["einc"].values] + einc_c3_cat + einc_c2_cat

gate_df["hhincome_w2"] = pd.Series(np.select(conditions, choices, default=np.nan))
# Convert to log income
gate_df["hhincome_w2"] = gate_df["hhincome_w2"].apply(
    lambda x: x if x == 0 else np.log(x)
)


# Create standardized measure of autonomy and risk-tolerance
gate_df["autonomy"] = abs(gate_df["sa_enjoys_working_independently"] - 6)
gate_df["autonomy_std"] = (
    gate_df["autonomy"] - gate_df["autonomy"].mean() / gate_df["autonomy"].std()
)
gate_df["risk_tolerance"] = (
    gate_df["sa_is_risk_averse"] + gate_df["sa_will_not_risk_savings"]
)
gate_df["risk_tolerance_std"] = (
    gate_df["risk_tolerance"]
    - gate_df["risk_tolerance"].mean() / gate_df["risk_tolerance"].std()
)


# Create final dataset
var_names = dataframes[3]["Final variables"].to_numpy()
gate_final = gate_df[gate_df.columns.intersection(var_names)].copy()


# Dummy variables to indicate whether missing value(s) are present
gate_cov = gate_final.drop(
    ["gateid", "completed_w2", "hhincome", "hhincome_w2", "site"], axis=1
)
gate_final["missing_cov"] = np.where(np.isnan(gate_cov).any(axis=1), 1, 0)
gate_final["missing_out"] = np.where(np.isnan(gate_final["hhincome_w2"]), 1, 0)


# Save gate_df as long dataset and gate_final as final dataset to work with
gate_df.to_csv(ppj("OUT_DATA", "gate_long.csv"), index=False)
gate_final.to_csv(ppj("OUT_DATA", "gate_final.csv"), index=False)
gate_final.dropna().to_csv(ppj("OUT_DATA", "gate_complete.csv"), index=False)
