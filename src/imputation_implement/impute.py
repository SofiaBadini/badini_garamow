"""Apply functions for imputation defined in ''imputation_method.py'' to get a
full sample for further analysis. The following data frames are created:

    **data_imputed_kNN** - imputed data frame, using the kNN imputation method.

    **data_imputed_kNN_msd** - imputed data frame, using first the ``msd``
    imputation method on the outcome and using the kNN imputation method on the
    covariates in the second step.

    **data_imputed_kNN_max** - imputed data frame, imputing first the mssings
    in the outcomes with the ``max`` of each column respectively and imputing the kNN imputation
    method on the missing values in the covariates in the second step.

    **data_imputed_kNN_min** - imputed data frame, imputing first the mssings
    in the outcomes with the ``min`` of each column respectively and imputing the kNN imputation
    method on the missing values in the covariates in the second step.

"""
import numpy as np
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.imputation_method.imputation_method import impute_kNN
from src.imputation_method.imputation_method import impute_msd

np.random.seed(42)

data = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Set the index to treatment and extrcat column names of columsn to be imputed.
data.set_index(keys=["treatment"], drop=False, inplace=True)

colname_outcome = ["hhincome_w2"]
colname_covariate = (
    data.select_dtypes(np.number).columns.difference(["gateid", "hhincome_w2"]).tolist()
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
    # Impute missings in outcome variables and covariates with kNN method.
    df[dict_colname["all"]] = impute_kNN(df, 1, dict_colname["all"])

for _key, df in dict_df_v2.items():
    # Impute the outcome variabels with the median and some standard error.
    df[dict_colname["outcome"]] = impute_msd(df, 1, 0.25, 0, dict_colname["outcome"])

for _key, df in dict_df_v3.items():
    # Impute the max of the column for the missings.
    df[dict_colname["outcome"]] = df[dict_colname["outcome"]].apply(
        lambda x: x.fillna(x.max(), axis=0, inplace=False)
    )

for _key, df in dict_df_v4.items():
    # Impute the min of the column for the missings.
    df[dict_colname["outcome"]] = df[dict_colname["outcome"]].apply(
        lambda x: x.fillna(x.min(), axis=0, inplace=False)
    )

for dict_df in dict_df_v2, dict_df_v3, dict_df_v3:
    # Impute the covariates with the kNN imputation method.
    for _key, df in dict_df.items():
        df[dict_colname["all"]] = impute_kNN(df, 1, dict_colname["all"])

data_imputed_kNN = pd.concat(dict_df_v1.values(), ignore_index=True).round(10)
data_imputed_kNN_msd = pd.concat(dict_df_v2.values(), ignore_index=True).round(10)
data_imputed_kNN_max = pd.concat(dict_df_v3.values(), ignore_index=True).round(10)
data_imputed_kNN_min = pd.concat(dict_df_v4.values(), ignore_index=True).round(10)

data_imputed_kNN.to_csv(ppj("OUT_IMPUTED_DATA", "data_imputed_kNN.csv"), index=False)
data_imputed_kNN_msd.to_csv(
    ppj("OUT_IMPUTED_DATA", "data_imputed_kNN_msd.csv"), index=False
)
data_imputed_kNN_max.to_csv(
    ppj("OUT_IMPUTED_DATA", "data_imputed_kNN_max.csv"), index=False
)
data_imputed_kNN_min.to_csv(
    ppj("OUT_IMPUTED_DATA", "data_imputed_kNN_min.csv"), index=False
)


for df in (
    data_imputed_kNN,
    data_imputed_kNN_msd,
    data_imputed_kNN_max,
    data_imputed_kNN_min,
):
    # Loop over data sets for completeness.
    if df.isna().sum().any(axis=0) is True:
        # See if data frame is complete.
        print("This data set is not complete.")
    else:
        print("Save complete data set.")
