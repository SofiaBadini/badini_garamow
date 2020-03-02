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
col_name = (
    data.select_dtypes(np.number).columns.difference(["gateid", "treatment"]).tolist()
)

data_dict_v1 = {}
data_dict_v2 = {}

for dict in data_dict_v1, data_dict_v2:
    """Create several dictionaries for treatment and control groups, which
    should be imputed later with different methods defined.

    """
    dict["treat"] = data.loc[data.treatment == 1].copy()
    dict["control"] = data.loc[data.treatment == 0].copy()

for _key, df in data_dict_v1.items():
    # Apply the imputation method to each treatment group respectively.
    df[col_name] = impute_msd(df, 1, 0.25, 0, col_name)

for _key, df in data_dict_v2.items():
    df[col_name] = impute_kNN(df, col_name)

data_imputed_v1 = pd.concat(data_dict_v1.values(), ignore_index=True)
data_imputed_v2 = pd.concat(data_dict_v2.values(), ignore_index=True)

for df in data_imputed_v1, data_imputed_v2:
    # Loop over data sets for completeness.
    if df.isna().sum().any(axis=0) is True:
        # See if data frame is complete.
        print("Data set not complete.")
    else:
        print("Save complete data set.")

df.to_csv(ppj("OUT_DATA", "data_imputed_v1.csv"), index=False)
df.to_csv(ppj("OUT_DATA", "data_imputed_v2.csv"), index=False)
