import numpy as np
import pandas as pd
import pytest
from impute_method import impute_kNN
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler

# from impute_method import impute_msd
# from pandas.testing import assert_series_equal

# from scr.method.define.impute_method import impute_kNN
# from scr.method.define.impute_method import impute_msd

scaler = StandardScaler()
nan = np.nan
np.random.seed(42)


@pytest.fixture
def setup_imputation():
    out = {}

    # case1: Standard case and k = 1.
    case1 = {
        "df": pd.DataFrame(
            np.array(
                [
                    [nan, 6, nan, 1, nan],
                    [4, 8, 5, 3, 1],
                    [nan, 0, 1, 0, 1],
                    [1, nan, 3, 0, nan],
                    [3, nan, 2, 0, 4],
                    [nan, 2, 3, 1, 0],
                ]
            ),
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 1,
        "col_name": ["A", "B", "C", "D", "E"],
    }
    out["case1"] = case1

    # case2: Standard case with k = 2.
    case2 = {
        "df": pd.DataFrame(
            np.array(
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
                ]
            ),
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 2,
        "col_name": ["A", "B", "C", "D", "E"],
    }
    out["case2"] = case2

    # case3: One row with all entries missing.
    case3 = {
        "df": pd.DataFrame(
            np.array(
                [
                    [nan, nan, nan, nan, nan],
                    [4, 8, 5, 3, 1],
                    [nan, 0, 1, 0, 1],
                    [1, nan, 3, 0, nan],
                    [3, nan, 2, 0, 4],
                    [nan, 2, 3, 1, 0],
                ]
            ),
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
            np.array([[nan, 6, nan, nan, nan], [4, 6, 5, 1, 1], [3, 6, 1, 0, 1]]),
            columns=["A", "B", "C", "D", "E"],
        ),
        "knn": 1,
        "col_name": ["A", "B", "C", "E"],
    }
    out["case4"] = case4

    return out


@pytest.fixture
def expected_imputation():
    out = {}

    case1 = {
        "df": pd.DataFrame(
            np.array(
                [
                    [3, 6, 3, 1, 0],
                    [4, 8, 5, 3, 1],
                    [1, 0, 1, 0, 1],
                    [1, 2, 3, 0, 0],
                    [3, 6, 2, 0, 4],
                    [1, 2, 3, 1, 0],
                ]
            ),
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case1"] = case1

    case2 = {
        "df": pd.DataFrame(
            np.array(
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
                ]
            ),
            columns=["A", "B", "C", "D", "E"],
        )
    }
    out["case2"] = case2

    case3 = {
        "df": pd.DataFrame(
            np.array(
                [
                    [2.666667, 3.333333, 2.8, 0.8, 1.5],
                    [4, 8, 5, 3, 1],
                    [1, 0, 1, 0, 1],
                    [1, 2, 3, 0, 0],
                    [3, 0, 2, 0, 4],
                    [1, 2, 3, 1, 0],
                ]
            ),
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


def test_kNN(setup_imputation, expected_imputation):
    """Test if the imputation of kNN imputer with k=1 is correct for standard
    case.

    """
    for case, expected_outcome in expected_imputation.items():
        calc_imputation = setup_imputation[case]["df"]
        calc_imputation[setup_imputation[case]["col_name"]] = impute_kNN(
            **setup_imputation[case]
        )
        calc_imputation = calc_imputation.round(10)

        assert_frame_equal(calc_imputation, expected_outcome["df"], check_dtype=False)
