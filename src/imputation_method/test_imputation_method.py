import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

from src.imputation_method.imputation_method import impute_kNN

scaler = StandardScaler()
nan = np.nan
np.random.seed(42)


@pytest.fixture
def setup_imputation_kNN():
    out = {}

    # case1: Standard case and k = 1.
    case1 = {
        "df": pd.DataFrame(
            [
                [nan, 6, nan, 1, nan],
                [4, 8, 5, 3, 1],
                [nan, 0, 1, 0, 1],
                [1, nan, 3, 0, nan],
                [3, nan, 2, 0, 4],
                [nan, 2, 3, 1, 0],
            ],
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 1,
        "col_name": ["A", "B", "C", "D", "E"],
    }
    out["case1"] = case1

    # case2: Standard case with k = 2.
    case2 = {
        "df": pd.DataFrame(
            [
                [nan, 3, 4, 1, nan],
                [4, nan, 5, 1, 1],
                [nan, 0, 1, 0, 1],
                [1, nan, 3, 0, nan],
                [3, nan, 2, 0, 4],
                [nan, 2, 3, 1, 0],
                [3, 2, nan, 0, 8],
                [6, 6, 3, 0, 2],
                [nan, 4, 3, 1, 3],
            ],
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 2,
        "col_name": ["A", "B", "C", "D", "E"],
    }
    out["case2"] = case2

    # case3: One row with all entries missing.
    case3 = {
        "df": pd.DataFrame(
            [
                [nan, nan, nan, nan, nan],
                [4, 8, 5, 3, 1],
                [nan, 0, 1, 0, 1],
                [1, nan, 3, 0, nan],
                [3, nan, 2, 0, 4],
                [nan, 2, 3, 1, 0],
            ],
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 1,
        "col_name": ["A", "B", "C", "D", "E"],
    }
    out["case3"] = case3

    # case4: Impute based on all columns except one and one column having the
    # same distance to all other columns.
    case4 = {
        "df": pd.DataFrame(
            [[nan, 6, nan, nan, nan], [4, 6, 5, 1, 1], [3, 6, 1, 0, 1]],
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 1,
        "col_name": ["A", "B", "C", "E"],
    }
    out["case4"] = case4

    return out


@pytest.fixture
def expected_imputation_kNN():
    out = {}

    case1 = {
        "df": pd.DataFrame(
            [
                [3, 6, 3, 1, 0],
                [4, 8, 5, 3, 1],
                [1, 0, 1, 0, 1],
                [1, 2, 3, 0, 0],
                [3, 6, 2, 0, 4],
                [1, 2, 3, 1, 0],
            ],
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case1"] = case1

    case2 = {
        "df": pd.DataFrame(
            [
                [3.5, 3, 4, 1, 1.5],
                [4, 2.5, 5, 1, 1],
                [2, 0, 1, 0, 1],
                [1, 1, 3, 0, 6],
                [3, 1, 2, 0, 4],
                [2.5, 2, 3, 1, 0],
                [3, 2, 2.5, 0, 8],
                [6, 6, 3, 0, 2],
                [5, 4, 3, 1, 3],
            ],
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case2"] = case2

    case3 = {
        "df": pd.DataFrame(
            [
                [2.666667, 3.333333, 2.8, 0.8, 1.5],
                [4, 8, 5, 3, 1],
                [1, 0, 1, 0, 1],
                [1, 2, 3, 0, 0],
                [3, 0, 2, 0, 4],
                [1, 2, 3, 1, 0],
            ],
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case3"] = case3

    case4 = {
        "df": pd.DataFrame(
            np.array([[4, 6, 5, nan, 1], [4, 6, 5, 1, 1], [3, 6, 1, 0, 1]]),
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case4"] = case4

    return out


def test_imputer_kNN(setup_imputation_kNN, expected_imputation_kNN):
    """Test the imputation of kNN imputer.

    """
    for case, expected_outcome in expected_imputation_kNN.items():
        calc_imputation = setup_imputation_kNN[case]["df"]
        calc_imputation[setup_imputation_kNN[case]["col_name"]] = impute_kNN(
            **setup_imputation_kNN[case]
        )
        calc_imputation = calc_imputation.round(10)

        assert_frame_equal(calc_imputation, expected_outcome["df"], check_dtype=False)


@pytest.fixture
def setup_imputation_msd():
    out = {}

    # case1: Imputing missings from a normal distribution around the mean with
    # the respective standard deviation times 0.25, of each column respectively.
    case1 = {
        "df": pd.DataFrame([[1, 2, np.nan], [np.nan, 3, 4], [5, np.nan, 6]]),
    }
    out["case1"] = case1

    return out


@pytest.fixture
def expected_imputation_msd():
    out = {}

    case1 = {
        "df": pd.DataFrame([[1, 2, 5], [3, 3, 4], [5, 2.5, 5]]),
    }
    out["case1"] = case1

    return out


def test_imputer_msd(setup_imputation_msd, expected_imputation_msd):
    """Test the imputation of msd imputer.

    """

    df_imputed = expected_imputation_msd["case1"]["df"]
    df_imputed_obs = df_imputed[setup_imputation_msd["case1"]["df"].isna()]

    df_to_test = []
    for i, j in zip(
        df_imputed_obs.items(), setup_imputation_msd["case1"]["df"].items()
    ):
        col_to_test = (i[1] <= j[1].median() + 0.25 * j[1].std()) & (
            i[1] >= j[1].median() - 0.25 * j[1].std()
        )
        df_to_test.append(col_to_test)

    expected_outcome_msd = pd.DataFrame(df_to_test).T
    calc_imputation_msd = pd.DataFrame(df_imputed_obs.notnull())

    assert_frame_equal(calc_imputation_msd, expected_outcome_msd, check_dtype=False)
