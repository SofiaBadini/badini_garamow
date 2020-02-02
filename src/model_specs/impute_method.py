"""Define methods to construct a complete data set on covariates or the outcome
variable(s) or both.

    *For covariates and outcome variable(s)*
    drop_any_missing(): Drop all rows with any missing value.

    impute_kNN(): Imputation of missings using the average kNN method.

    *For covariates only*
    impute_median(): Impute missings by the median of each column.

    *For outcome variable(s) only*
    impute_median(): Impute missings by the median of each column,
    plus/ minus some standard deviation.

"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def drop_any_missing(data):
    """Drop all rows with any missing value.

        Inputs:
        data (pd.DataFrame)

        Returns:
        complete data set, df_complete (pd.DataFrame)

    """
    df_complete = data.dropna(axis=0)

    return df_complete

def impute_kNN(data, k):
    """Impute missing values with the average of the k nearest neighbors based
    on the covariates from the baseline.

        Inputs:
        data (pd.DataFrame)

        Returns:
        imputed data set, df_imputed_kNN (pd.DataFrame)

    """
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    df_imputed_kNN = imputer.fit_transform(data)

    return df_imputed_kNN


def impute_median(data):
    """Impute missing values with the median of ech column.

        Inputs:
        data (pd.DataFrame)

        Returns:
        imputed data set, df_imputed_median (pd.DataFrame)

    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df_imputed_median = imputer.fit_transform(data)

    return df_imputed_median

def impute_mean_sd(data, sd):
    """Impute the missing values with the median, plus/ minus some standard
    deviation.

    """
