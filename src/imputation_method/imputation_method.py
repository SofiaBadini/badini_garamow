import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

nan = np.nan
scaler = StandardScaler()


def impute_msd(df, k, sd_share, sd_fixed, col_name):
    """Impute the missing values with the median, plus/ minus some share of the
    standard deviation of the specific variable. Please note that the data frame
    is divided

    Args:

    df (pd.DataFrame() - data set
    k (integer) - number of draws
    sd_share (integer) - share of variance applied
    sd_fixed (integer) - additional constant to variance
    col_name (list) - names of variables which should be imputed

    Returns:

    colmiss_ndarray (nd array) - imputed nd array

    """
    col_ndarray = df[col_name].values

    col_len = np.arange(np.size(col_ndarray, 1))
    col_median = np.nanmedian(col_ndarray, axis=0)
    col_sd = np.nanstd(col_ndarray, axis=0)

    for i in col_len:
        col_nan_placeholder = np.isnan(col_ndarray[:, i])
        col_fill = np.random.normal(
            col_median[i],
            col_sd[i] * sd_share + sd_fixed,
            [col_nan_placeholder.sum(), k],
        ).mean(axis=1)
        col_ndarray[:, i][col_nan_placeholder] = col_fill

    return col_ndarray


def impute_kNN(df, knn, col_name):
    """Impute the missing values with the average of the k nearest neightbors.

    Args:

    df (pd.DataFrame() - data set
    knn (integer) - number of nearest neighbors
    col_name (list) - names of variables which should be imputed

    Returns:

    col_ndarray_inverse () - imputed nd array

    """
    imputer_kNN = KNNImputer(n_neighbors=knn, missing_values=nan, weights="uniform")
    col_ndarray = df[col_name].values
    col_ndarray_normlized = scaler.fit_transform(col_ndarray)
    col_ndarray_imputed = imputer_kNN.fit_transform(col_ndarray_normlized)
    col_ndarray_inverse = scaler.inverse_transform(col_ndarray_imputed)

    return col_ndarray_inverse
