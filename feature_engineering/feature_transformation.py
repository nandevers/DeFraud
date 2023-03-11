import numpy as np
import pandas as pd
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRD_SAMP_FEAT_DATA = "data/processed/psr_sampled_features.csv"
data = pd.read_csv(PRD_SAMP_FEAT_DATA)
data.shape
data.columns
CUTOFF_YEAR = "2020"

ID_COLS = [
    "nr_documento_segurado",
    "nr_proposta",
    "id_proposta",
    "dt_proposta",
    "evento_preponderante",
]
STD_COLS = [
    "valor_indenizacao",
    # "nr_previous_['dt_proposta','id_proposta']",
    "max_nr_area_total",
    "min_nr_area_total",
    "sum_nr_area_total",
    "max_niveldecobertura",
    "min_niveldecobertura",
    "sum_niveldecobertura",
    "max_valor_indenizacao",
    "min_valor_indenizacao",
    "sum_valor_indenizacao",
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


def handle_missing_values(data, method):
    data["evento_preponderante"].fillna("outro", inplace=True)
    mmi = MeanMedianImputer(imputation_method=method)
    mmi.fit(data.loc[data.dt_proposta <= CUTOFF_YEAR, STD_COLS[1:]])
    data.loc[:, STD_COLS[1:]] = mmi.transform(data.loc[:, STD_COLS[1:]])
    return data


def one_hot_encode(data):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist")
    ep_train = data.loc[data.dt_proposta <= CUTOFF_YEAR, "evento_preponderante"]
    ep = data["evento_preponderante"]
    enc.fit(np.reshape(ep_train.values, (-1, 1)))
    ep_matrix = enc.transform(np.reshape(ep.values, (-1, 1))).toarray()
    ep_matrix = pd.DataFrame(ep_matrix)
    ep_cols = [f"evento_preponderante_{c}" for c in ep_matrix.columns]
    ep_matrix.columns = ep_cols
    data = pd.concat([data, ep_matrix], axis=1)
    return data, ep_cols


def standard_scaler(data):
    std_new_cols = [f"std_{c}" for c in STD_COLS]
    scaler = StandardScaler()
    scaler.fit(data.loc[data.dt_proposta <= CUTOFF_YEAR, STD_COLS])
    std_data = scaler.transform(data.loc[:, STD_COLS])
    std_data = pd.DataFrame(std_data, columns=std_new_cols)
    data = pd.concat([data, std_data], axis=1)
    return data, std_new_cols


data[STD_COLS[1:]]

data = handle_missing_values(data, "median")
data, ep_cols = one_hot_encode(data)
data, std_cols = standard_scaler(data)

cols_to_write = ID_COLS + std_cols + ep_cols + ["valor_indenizacao"]
data.loc[data.dt_proposta <= CUTOFF_YEAR, cols_to_write].to_csv(
    "data/processed/psr_train_set.csv", index=False
)

data.loc[data.dt_proposta > CUTOFF_YEAR, cols_to_write].to_csv(
    f"data/processed/psr_test_set.csv", index=False
)

data.loc[:, cols_to_write].to_csv(
    f"data/processed/psr_train_and_test_set.csv", index=False
)
