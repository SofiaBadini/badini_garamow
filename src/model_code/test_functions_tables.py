import numpy as np
import pandas as pd
import pytest
from functions_tables import chisquare_by_column
from functions_tables import compute_sample_sizes
from functions_tables import create_quantile_dummy
from functions_tables import levene_by_column
from functions_tables import ttest_by_column
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
    out["df_cont"] = pd.DataFrame(data_continuous, columns=list("ABC"))
    out["dummy"] = "C"
    out["equal_var"] = False
    out["df_alldummy"] = pd.DataFrame(data_dummy, columns=list("ABC"))
    out["df_fordummy"] = pd.DataFrame(
        data=[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        columns=["values", "zeroes"],
        index=list("abcde"),
    )
    out["values"] = "values"
    out["median"] = True
    return out


@pytest.fixture
def expected_dataframes():
    out = {}
    out["student"] = pd.DataFrame(
        data=[[-0.09419, -0.13376, 0.81917], [0.13977, -0.05606, 0.33101]],
        columns=["mean0", "mean1", "pvalue"],
        index=list("AB"),
    )
    out["welch"] = pd.DataFrame(
        data=[[-0.09419, -0.13376, 0.81916], [0.13977, -0.05606, 0.32800]],
        columns=["mean0", "mean1", "pvalue"],
        index=list("AB"),
    )
    out["levene"] = pd.DataFrame(
        data=[[0.0003182, 0.98580], [0.71753, 0.39901]],
        columns=["levene_stat", "pvalue"],
        index=list("AB"),
    )
    out["chisquare"] = pd.DataFrame(
        data=[[0.94457, 0.33110], [0.02156, 0.88324]],
        columns=["chisq", "pvalue"],
        index=list("AB"),
    )
    out["sample_size"] = [
        pd.Series(46, index=["sample_size"]),
        pd.Series(54, index=["sample_size"]),
    ]
    out["dummy_quantiles"] = pd.DataFrame(
        data=[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        columns=["values_p25", "values_p25_75", "values_p75"],
        index=list("abcde"),
    )
    out["dummy_median"] = pd.Series(
        data=[1, 1, 1, 0, 0], name="low_values", index=list("abcde")
    )
    return out


# Unit test for ttest_by_column (classic Student's t-test)
def test_ttest_by_column_student(setup, expected_dataframes):
    calc_frame = ttest_by_column(setup["df_cont"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["student"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for ttest_by_column (Welch's test)
def test_ttest_by_column_welch(setup, expected_dataframes):
    calc_frame = ttest_by_column(setup["df_cont"], setup["dummy"], setup["equal_var"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["welch"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for levene_by_column
def test_levene_by_column(setup, expected_dataframes):
    calc_frame = levene_by_column(setup["df_cont"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["levene"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit tests for chisquare_by_column
def test_chisquare_by_column(setup, expected_dataframes):
    calc_frame = chisquare_by_column(setup["df_alldummy"], setup["dummy"])
    assert_frame_equal(
        calc_frame,
        expected_dataframes["chisquare"],
        check_exact=False,
        check_less_precise=True,
    )


# Unit test for compute_sample_sizes
def test_compute_sample_sizes(setup, expected_dataframes):
    calc_frame = compute_sample_sizes(setup["df_cont"], setup["dummy"])
    for i in (0, 1):
        assert_series_equal(
            calc_frame[i],
            expected_dataframes["sample_size"][i],
            check_exact=False,
            check_less_precise=True,
        )


# Unit tests for create_quantile_dummy
def test_create_quantile_dummy(setup, expected_dataframes):
    calc_frame = create_quantile_dummy(setup["df_fordummy"], setup["values"])
    assert_frame_equal(
        calc_frame, expected_dataframes["dummy_quantiles"], check_dtype=False
    )


def test_create_quantile_dummy_median(setup, expected_dataframes):
    calc_frame = create_quantile_dummy(
        setup["df_fordummy"], setup["values"], setup["median"]
    )
    assert_series_equal(
        calc_frame, expected_dataframes["dummy_median"], check_dtype=False
    )
