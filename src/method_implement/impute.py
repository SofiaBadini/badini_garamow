"""Apply functions for imputation defined in ''method_define.py'' to get a
full sample for further analysis.

"""
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj

from src.method_define.impute_method import impute_kNN
from src.method_define.impute_method import impute_msd


data = pd.read_csv(ppj("IN_DATA", "gate_final.csv"))

# Set the index to treatment and extrcat column names of columsn to be imputed.
data.set_index(keys=["treatment"], drop=False, inplace=True)

colname_outcome = ["hhincome_w2"]
colname_covariate = (
    data.select_dtypes(np.number)
    .columns.difference(["treatment", "gateid", "hhincome_w2"])
    .tolist()
)
colname_all = colname_outcome + colname_covariate
dict_colname = {
    "outcome": colname_outcome,
    "covariate": colname_covariate,
    "all": colname_all,
}

dict_df_v1 = {}
dict_df_v2 = {}
dict_df_v3 = {}
dict_df_v4 = {}

dict_df_list = [dict_df_v1, dict_df_v2, dict_df_v3, dict_df_v4]

for dict in dict_df_list:
    # Divide the data set into treatment and control groups in the dictionary.
    dict["treat"] = data.loc[data.treatment == 1].copy()
    dict["control"] = data.loc[data.treatment == 0].copy()

for _key, df in dict_df_v1.items():
    # important to round!!!
    df[dict_colname["all"]] = impute_kNN(df, dict_colname["all"])

for dict_df in dict_df_v2, dict_df_v3, dict_df_v4:
    # Loop over the dictionareis _v2, _v3 and _v4.
    for _key, df in dict_df.items():
        # Impute the missing values in the covariates with the kNN imputer.
        df[dict_colname["covariate"]] = impute_kNN(df, dict_colname["all"])

for _key, df in dict_df_v2.items():
    # Impute the missing values in the outcome with the msd imputer.
    df[dict_colname["outcome"]] = impute_msd(df, 1, 0.25, 0, dict_colname["outcome"])

for _key, df in dict_df_v3.items():
    # Impute the missing values in the outcome with the column minimum.
    df[dict_colname["outcome"]] = df[dict_colname["outcome"]].apply(
        lambda x: x.fillna(x.min(), axis=0, inplace=False)
    )

for _key, df in dict_df_v4.items():
    # Impute the missing values in the outcome with the column maximum.
    df[dict_colname["outcome"]] = df[dict_colname["outcome"]].apply(
        lambda x: x.fillna(x.max(), axis=0, inplace=False)
    )

data_imputed_v1 = pd.concat(dict_df_v1.values(), ignore_index=True).round(10)
data_imputed_v2 = pd.concat(dict_df_v2.values(), ignore_index=True).round(10)
data_imputed_v3 = pd.concat(dict_df_v3.values(), ignore_index=True).round(10)
data_imputed_v4 = pd.concat(dict_df_v4.values(), ignore_index=True).round(10)

for df in (
    data_imputed_v1,
    data_imputed_v2,
    data_imputed_v3,
    data_imputed_v4,
):
    # Loop over data sets for completeness.
    if df.isna().sum().any(axis=0) is True:
        # See if data frame is complete.
        print("Data set not complete.")
    else:
        print("Save complete data set.")

df.to_csv(ppj("OUT_DATA", "data_imputed_v1.csv"), index=False)
df.to_csv(ppj("OUT_DATA", "data_imputed_v2.csv"), index=False)
df.to_csv(ppj("OUT_DATA", "data_imputed_v3.csv"), index=False)
df.to_csv(ppj("OUT_DATA", "data_imputed_v4.csv"), index=False)
