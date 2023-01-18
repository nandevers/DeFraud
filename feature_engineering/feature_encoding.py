import pandas as pd
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="infrequent_if_exist")

PRD_SAMP_FEAT_DATA = "data/processed/psr_sampled_featues.csv"
data = pd.read_csv(PRD_SAMP_FEAT_DATA)

data.isna().sum()


data["evento_preponderante"] = data["evento_preponderante"].fillna("outro")
im = ["median", "mean"]
for i in im:
    mmi = MeanMedianImputer(imputation_method=i)
    mmi.fit(data.loc[data.dt_proposta <= "2013"])
    data = mmi.transform(data)

ep_train = data.loc[data.dt_proposta <= "2013", "evento_preponderante"]
ep = data["evento_preponderante"]


import numpy as np


enc.fit(np.reshape(ep_train.values, (-1, 1)))

ep_matrix = enc.transform(np.reshape(ep.values, (-1, 1))).toarray()
ep_matrix = pd.DataFrame(ep_matrix)
ep_cols = [f"evento_preponderante_{c}" for c in ep_matrix.columns]
ep_matrix.columns = ep_cols
data = pd.concat([data, ep_matrix], axis=1)

std_cols = [
    "valor_indenização",
    "nr_previous_dt_proposta",
    "max_nr_area_total",
    "min_nr_area_total",
    "sum_nr_area_total",
    "max_niveldecobertura",
    "min_niveldecobertura",
    "sum_niveldecobertura",
    "max_valor_indenização",
    "min_valor_indenização",
    "sum_valor_indenização",
    "max_vl_subvencao_federal",
    "min_vl_subvencao_federal",
    "sum_vl_subvencao_federal",
    "max_pe_taxa",
    "min_pe_taxa",
    "sum_pe_taxa",
    "max_vl_premio_liquido",
    "min_vl_premio_liquido",
    "sum_vl_premio_liquido",
    "max_nr_produtividade_segurada",
    "min_nr_produtividade_segurada",
    "sum_nr_produtividade_segurada",
    "max_vl_limite_garantia",
    "min_vl_limite_garantia",
    "sum_vl_limite_garantia",
]

std_col_names = [f"std_{c}" for c in std_cols]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data.loc[data.dt_proposta <= "2013", std_cols])
std_data = scaler.transform(data.loc[:, std_cols])
std_data = pd.DataFrame(std_data, columns=std_col_names)
data = pd.concat([data, std_data], axis=1)


ids = ["nr_documento_segurado", "nr_proposta", "dt_proposta", "evento_preponderante"]
cols_to_write = ids + std_col_names + ep_cols + ["valor_indenização"]


data.loc[data.dt_proposta <= "2013", cols_to_write].to_csv(
    f"data/processed/psr_{i}_train_set.csv", index=False
)
data.loc[data.dt_proposta > "2013", cols_to_write].to_csv(
    f"data/processed/psr_{i}_test_set.csv", index=False
)
