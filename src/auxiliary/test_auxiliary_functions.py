import numpy as np
import pandas as pd
import pytest
from auxiliary_functions import chisquare_by_column
from auxiliary_functions import create_quantile_dummy
from auxiliary_functions import generate_regression_output
from auxiliary_functions import levene_by_column
from auxiliary_functions import ttest_by_column
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

np.random.seed(42)

# Generate data
continuous = np.random.normal(0, 1, size=(100, 2))
dummy = np.random.choice([0, 1], size=(100, 1), p=[0.5, 0.5])
data_continuous = np.concatenate((continuous, dummy), axis=1)

# Generate data
dummies = np.random.choice([0, 1], size=(100, 2), p=[0.5, 0.5])
data_dummy = np.concatenate((dummies, dummy), axis=1)


@pytest.fixture
def setup():
    out = {}
    out["regressand"] = "A"
    out["dummy"] = "C"
    out["type"] = "Logit"
    out["values"] = "values"
    out["df_continuous"] = pd.DataFrame(data_continuous, columns=list("ABC"))
    out["df_dummies"] = pd.DataFrame(data_dummy, columns=list("ABC"))
    out["df_quantile"] = pd.DataFrame(
        data=[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        columns=["values", "zeroes"],
        index=list("abcde"),
    )
    out["median"] = True
    out["equal_var"] = False
    return out


@pytest.fixture
def expected_dataframes():
    out = {}
    out["student"] = pd.DataFrame(
        data=[[-0.09419, -0.13376, 0.81917], [0.13977, -0.05606, 0.33101]],
        columns=["mean0", "mean1", "p-value"],
        index=list("AB"),
    )
    out["welch"] = pd.DataFrame(
        data=[[-0.09419, -0.13376, 0.81916], [0.13977, -0.05606, 0.32800]],
        columns=["mean0", "mean1", "p-value"],
        index=list("AB"),
    )
    out["levene"] = pd.DataFrame(
        data=[[0.0003182, 0.98580], [0.71753, 0.39901]],
        columns=["test stat.", "p-value"],
        index=list("AB"),
    )
    out["chisquare"] = pd.DataFrame(
        data=[[0.94457, 0.33110], [0.02156, 0.88324]],
        columns=["test stat.", "p-value"],
        index=list("AB"),
    )
    out["dummy_quantiles"] = pd.DataFrame(
        data=[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        columns=["values_p25", "values_p25_75", "values_p75"],
        index=list("abcde"),
    )
    out["dummy_median"] = pd.Series(
        data=[1, 1, 1, 0, 0], name="low_values", index=list("abcde")
    )
    out["df_OLS_coeff"] = pd.DataFrame(
        data=[[-0.0978, 0.128, 0.445], [0.0260, 0.069, 0.708], [-0.0345, 0.174, 0.843]],
        columns=["Coeff.", "Std. Error", "p-value"],
        index=["constant", "B", "C"],
    )
    out["df_OLS_stats"] = pd.DataFrame(
        data=[100, 0.001447, -0.019141, 0.096418],
        columns=["Summary statistics"],
        index=["Observations", "R-squared", "Adjusted R-squared", "F Statistic"],
    )
    out["df_logit"] = pd.DataFrame(
        data=[[0.1631, 0.204, 0.423], [-0.0471, 0.236, 0.842], [-0.1984, 0.204, 0.332]],
        columns=["Coeff.", "Std. Error", "p-value"],
        index=["constant", "A", "B"],
    )
    return out


# Unit test for ttest_by_column (classic Student's t-test)
def test_ttest_by_column_student(setup, expected_dataframes):
    calc_frame = ttest_by_column(setup["df_continuous"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["student"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for ttest_by_column (Welch's test)
def test_ttest_by_column_welch(setup, expected_dataframes):
    calc_frame = ttest_by_column(
        setup["df_continuous"], setup["dummy"], setup["equal_var"]
    )
    assert_frame_equal(
        calc_frame,
        expected_dataframes["welch"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for levene_by_column
def test_levene_by_column(setup, expected_dataframes):
    calc_frame = levene_by_column(setup["df_continuous"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["levene"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit tests for chisquare_by_column
def test_chisquare_by_column(setup, expected_dataframes):
    calc_frame = chisquare_by_column(setup["df_dummies"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["chisquare"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit tests for create_quantile_dummy (median=False)
def test_create_quantile_dummy(setup, expected_dataframes):
    calc_frame = create_quantile_dummy(setup["df_quantile"], setup["values"])
    assert_frame_equal(
        calc_frame, expected_dataframes["dummy_quantiles"], check_dtype=False
    )


# Unit tests for create_quantile_dummy (median=True)
def test_create_quantile_dummy_median(setup, expected_dataframes):
    calc_frame = create_quantile_dummy(
        setup["df_quantile"], setup["values"], setup["median"]
    )
    assert_series_equal(
        calc_frame, expected_dataframes["dummy_median"], check_dtype=False
    )


# Unit test for generate_regression_output (OLS, first dataframe)
def test_generate_regression_output_OLS1(setup, expected_dataframes):
    calc_frame = generate_regression_output(setup["df_continuous"], setup["regressand"])
    assert_frame_equal(calc_frame[0], expected_dataframes["df_OLS_coeff"])


# Unit test for generate_regression_output (OLS, second dataframe)
def test_generate_regression_output_OLS2(setup, expected_dataframes):
    calc_frame = generate_regression_output(setup["df_continuous"], setup["regressand"])
    assert_frame_equal(
        calc_frame[1],
        expected_dataframes["df_OLS_stats"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for generate_regression_output (logistic)
def test_generate_regression_output_logit(setup, expected_dataframes):
    calc_frame = generate_regression_output(
        setup["df_continuous"], setup["dummy"], setup["type"]
    )
    assert_frame_equal(calc_frame, expected_dataframes["df_logit"])
