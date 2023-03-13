import json
import pandas as pd
from pandas_profiling import ProfileReport
import streamlit as st


ID_COLS = [
    "nr_documento_segurado",
    "nr_proposta",
    "dt_proposta",
    "evento_preponderante",
]
STD_COLS = [
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

st.set_page_config(layout="wide")


def read_cols():
    with open("cols_dict.json") as f:
        cols_dict = json.load(f)
    cols_dict["value_columns"].append("valor_indenização")
    return cols_dict


ID = "nr_documento_segurado"

@st.cache_data 
def read_sample():
    return pd.read_csv("data/processed/psr_soja_pr_sample.csv")


@st.cache_resource
def read_features():
    return pd.read_csv("data/processed/psr_sampled_features.csv")


@st.cache_resource
def read_tfeatures():
    return pd.read_csv("data/processed/psr_train_and_test_set.csv").query('valor_indenizacao==valor_indenizacao')


def profile_features():
    yield ProfileReport(read_features()).to_file("assets/profile_features.html")


st.markdown(
    """
    ## Data pipeline

    **In this section, you'll find the following datasets:**




"""
)

with st.sidebar:
    sample = read_sample()
    st.session_state.selected_id = st.selectbox(
        f"Select sample by {ID}",
        options=sample[ID].value_counts().sort_values(ascending=False).index,
    )
    st.session_state.feature = st.sidebar.selectbox(
        "Select feature:", sorted(sample.columns.to_list())
    )


    if st.sidebar.button("Clear All"):
        # Clears all st.cache_resource caches:
        st.cache_resource.clear()


raw_t, feature_t, tr_feature_t, results_t = st.tabs(
    ["Raw", "Features", "Transformation", "Results"]
)
with raw_t:
    
    st.markdown(
        """
        Original dataset obtained from official authorities in Brazil. 
        """
    )
    
    selected_case_raw_df = sample[sorted(sample.columns)].query(
        f"{ID}=='{st.session_state.selected_id}'"
    )
    if "selected_id" in st.session_state:
        selected_case_feat_df = read_features().query(
            f"{ID}=='{st.session_state.selected_id}'"
        )
    
    if "selected_id" in st.session_state:
        raw_cols = read_cols()["code_id_cols"]
        raw_cols.append("dt_proposta")
        st.dataframe(selected_case_raw_df[raw_cols])

    

with feature_t:
    
    st.markdown(
        """
        This is the first extract of features. They do not depend on the sampling strategy for training. 
        """
    )
    features = selected_case_feat_df.columns[
        selected_case_feat_df.columns.str.contains(st.session_state.feature)
    ].to_list()
    features.remove(st.session_state.feature)
    features = [ID] + features
    raw_cols = list(set(raw_cols))

    if "id_proposta" != st.session_state.feature:
        raw_cols.append(st.session_state.feature)
        features.insert(0, "id_proposta")

    selected_case_feat_df = pd.merge(
        selected_case_raw_df.loc[:, raw_cols],
        selected_case_feat_df.loc[:, features + ["dt_proposta"]],
        on=[ID, "dt_proposta", "id_proposta"],
    )

    st.write("ID: ", selected_case_feat_df[ID].unique()[0])
    st.dataframe(selected_case_feat_df.sort_values(["dt_proposta", "id_proposta"]))

with tr_feature_t:
    
    st.markdown(
        """
        Sampling dependent transformations are applied to get the final input data set which is fed through the environment.
        """
    )
    
 



    if "selected_id" in st.session_state:
        selected_case_tfeat_df = read_tfeatures().query(
            f"{ID}=='{st.session_state.selected_id}'"
        )
        st.dataframe(selected_case_tfeat_df)
