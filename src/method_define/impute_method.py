"""Define methods to construct a complete data set on covariates or the outcome
variable(s) or both.

    *For outcome variable(s) and covariates*
    impute_kNN(): Imputation of missings using the average kNN method.

    impute_median(): Impute missings by the median of each column,
    plus/ minus some standard deviation.

"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def impute_kNN(df, k):
    """Impute missing values with the average of the k nearest neighbors based
    on the covariates from the baseline.

        Args:

            df: Data frame (pd.DataFrame)

            k: Number of neighbors (integer)

        Out:

            df_imputed_kNN: Imputed data set, df_imputed_kNN (pd.DataFrame)

    """
    imputer = KNNImputer(n_neighbors=k, weights="uniform")
    df_imputed_kNN = imputer.fit_transform(df)

    return df_imputed_kNN

def impute_median_sd(col, sd_share, sd_fixed, k):
    """Impute the missing values with the median, plus/ minus some share of the
    standard deviation of the specific variable.

        Args:

            col: Column/ series of the respective data frame (PandaSeries)

            sd_share: Share of the variance (integer)

            sd_fixed: Add a constant to the variance (integer)

            k: number of imputations (integer)

        Out:

            col: Column with imputed values (PandaSeries)

    """
    if is_numeric_dtype(col)==True:

        col_array = col.values
        colmiss_nan_placeholder = np.isnan(col_array)

        if colmiss_nan_placeholder.any()==True and np.ma.all(colmiss_nan_placeholder)==False:

            colmiss_median = np.nanmedian(col_array, axis=0)
            colmiss_sd = np.nanstd(col_array, axis=0)

            colmiss_fill = np.random.normal(colmiss_median, colmiss_sd*sd_share+sd_fixed, [colmiss_nan_placeholder.sum(), k]).mean(axis=1)
            col_array[colmiss_nan_placeholder] = colmiss_fill
            col = col_array

        else:
            print('No or all missing entries')
            col = col

    else:
        print('Not numeric type')
        col = col

    return col


def impute_median_sd_2(df, k, outvar):
    """Impute the missing values with the median, plus/ minus some share of the
    standard deviation of the specific variable.

    """

    colmiss_name = df[outvar].select_dtypes(np.number).columns.tolist()
    colmiss_ndarray = df[name].values

    colmiss_len = np.arange(np.size(colmiss_ndarray, 1))
    colmiss_median = np.nanmedian(colmiss_ndarray, axis = 0)
    colmiss_sd = np.nanstd(colmiss_ndarray, axis=0)

    for i in colmiss_len:
        colmiss_nan_placeholder = np.isnan(colmiss_ndarray[:, i])
        colmiss_fill = np.random.normal(colmiss_median[i], colmiss_sd[i], [colmiss_nan_placeholder.sum(), k]).mean(axis=1)
        colmiss_ndarray[:,i][colmiss_nan_placeholder] = colmiss_fill

    df_filled = df
    df_filled[name] = colmiss_ndarray

    return df_filled
