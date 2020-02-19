"""Here we impute the missing values/ observations."""

from src.method_define.impute_method import impute_kNN
from src.method_define.impute_method import impute_median_sd

variable_name = {
  'outcome': 'population',
  'covariate': ['GDP', 'alpha-2'],
}

imputation_variation= {
  'variation1': {'function': impute_median_sd, 'sd_share': 0.10, 'sd_fixed': 0, 'k': 1},
  'variation2': {'function': impute_median_sd, 'sd_share': 0.25, 'sd_fixed': 0, 'k': 1},
  'variation3': {'function': impute_median_sd, 'sd_share': 0.75, 'sd_fixed': 0, 'k': 1},
  'variation4': {'function': impute_median_sd, 'sd_share': 0.10, 'sd_fixed': 0, 'k': 10},
  'variation5': {'function': impute_median_sd, 'sd_share': 0.25, 'sd_fixed': 0, 'k': 10},
  'variation5': {'function': impute_median_sd, 'sd_share': 0.75, 'sd_fixed': 0, 'k': 10}
}

# Or directly variable_name['outcome']
outcome_name = variable_name['outcome']
covariate_name = variable_name['covariate']

# Impute outcomes
df_outcome_imputed_kNN = impute_kNN(OURDATA[outcome_name], k=1)

df_outcome_imputed_median_sd = OURDATA[outcome_name].apply(
                            impute_median_sd, sd_share=0.25, sd_fixed=0, k=1
                            )

# Impute covariates
df_covariate_imputed_kNN = impute_kNN(OURDATA[covariate_name], k=1)
