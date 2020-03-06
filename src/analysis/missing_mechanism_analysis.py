"""Analysis of missing data mechanism in ``gate_final.csv``, stored in the
"OUT_DATA" directory. Results are saved to .csv files and stored in the
"OUT_ANALYSIS" directory.

"""
import pandas as pd

from bld.project_paths import project_paths_join as ppj
from src.analysis.analysis_functions import create_chisq_dataframe
from src.analysis.analysis_functions import create_integrity_dataframe
from src.analysis.analysis_functions import create_levene_dataframe
from src.analysis.analysis_functions import create_logistic_dataframe
from src.analysis.analysis_functions import create_welch_dataframe


gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))
create_chisq_dataframe()
create_integrity_dataframe()
create_levene_dataframe()
create_logistic_dataframe()
create_welch_dataframe()
