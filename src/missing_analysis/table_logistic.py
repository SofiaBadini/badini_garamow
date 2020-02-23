"""Check for missing mechanism via logistic regressions.

Analyse data in ``gate_final.csv``, stored in the "OUT_DATA" directory, and save
tables to ``table_legistic.tex`` in the "OUT_TABLES" directory.

"""
import pandas as pd
import statsmodels.api as sm

from bld.project_paths import project_paths_join as ppj
from src.missing_analysis.formatting_tables import assign_stars
from src.missing_analysis.pretty_index import pretty_index_dict


# Load dataset
gate_final = pd.read_csv(ppj("OUT_DATA", "gate_final.csv"))

# Drop redundant variables dataset
gate_logit = gate_final.drop(
    [
        "gateid",
        "hhincome",
        "hhincome_w2",
        "completed_w2",
        "missing_out",
        "site",
        "philadelphia",
        "white",
        "hhincome_p50_74",
        "agesqr",
        "worked_for_relatives_friends_se",
    ],
    axis=1,
)

# Store indexes
index_cov = ["constant"] + list(gate_logit.drop(["missing_cov"], axis=1).columns)
index_out = ["constant"] + list(gate_logit.columns)

# Logistic regression for missing in covariates
y_cov = gate_final["missing_cov"].values
x_cov = gate_logit.drop(["missing_cov"], axis=1).values
x_cov = sm.add_constant(x_cov)
model_cov = sm.Logit(y_cov, x_cov, missing="drop").fit(disp=False)
results_cov = model_cov.summary().tables[1].as_html()

# Logistic regression for missing in outcome
y_out = gate_final["missing_out"].values
x_out = gate_logit.values
x_out = sm.add_constant(x_out)
model_out = sm.Logit(y_out, x_out, missing="drop").fit(disp=False)
results_out = model_out.summary().tables[1].as_html()

# Recreate results' DataFrames
results_cov = pd.read_html(results_cov, header=0, index_col=0)[0]
results_out = pd.read_html(results_out, header=0, index_col=0)[0]

# Rename indexes
dict_cov = dict(zip(results_cov.index, index_cov))
results_cov = results_cov.rename(index=dict_cov)
dict_out = dict(zip(results_out.index, index_out))
results_out = results_out.rename(index=dict_out)

# Create MultiIndex DataFrame
table_logistic = pd.concat(
    [results_cov, results_out],
    axis=1,
    keys=["Missing in covariates", "Missing in outcome"],
    sort=False,
)

# Format DataFrame
table_logistic = table_logistic.rename(
    columns=({"coef": "Coeff.", "std err": "Std. Error", "P>|z|": "p-value"})
)
table_logistic = table_logistic.reindex(pretty_index_dict).rename(pretty_index_dict)
table_logistic = table_logistic.dropna(how="all")
idx = pd.IndexSlice
table_logistic = table_logistic.loc[
    idx[:, idx[:, ["Coeff.", "Std. Error", "p-value"]]]
].copy()
subset = idx[:, idx[:, "p-value"]]
table_logistic = assign_stars(table_logistic, subset)
table_logistic = table_logistic.fillna(" ")
table_logistic

# Save to latex table
table_logistic.to_latex(
    ppj("OUT_TABLES", "table_logistic.tex"),
    float_format="{:.3g}".format,
    na_rep=" ",
    multicolumn_format="c",
)
