"""Perform OLS regressions to estimate ITT on datasets stored in the "OUT_DATA"
and in the "OUT_IMPUTED_DATA" directory. Results are saved to .csv files and stored
in the "OUT_ANALYSIS" directory.

"""
from src.analyis.analysis_functions import itt_analysis_with_controls
from src.analyis.analysis_functions import itt_analysis_without_controls


dict_data_list = {
    "gate_complete": "OUT_DATA",
    "data_imputed_kNN": "OUT_IMPUTED_DATA",
    "data_imputed_kNN_msd": "OUT_IMPUTED_DATA",
    "data_imputed_kNN_max": "OUT_IMPUTED_DATA",
    "data_imputed_kNN_min": "OUT_IMPUTED_DATA",
}

for key, in_dir in dict_data_list.items():

    itt_analysis_with_controls(key, in_dir)
    itt_analysis_without_controls(key, in_dir)
