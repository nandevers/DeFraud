# Features
import streamlit as st
import pandas as pd

raw_df = pd.read_csv("data/processed/psr_soja_pr_sample.csv", nrows=1000)
customers = raw_df["nr_documento_segurado"]
features = raw_df.columns.to_list()


tab0, tab1, tab2, tab3 = st.tabs(["Raw", "Extraction", "Engineering", "CheckPoint"])


with st.sidebar:
    selected = st.selectbox(label="Select a customer", options=customers, index=0)

with tab0:
    customer_data = raw_df[raw_df["nr_documento_segurado"] == selected].sort_values(
        "nr_proposta"
    )
    st.write("**Customer Information**")
    st.subheader(customer_data.nm_segurado.iloc[0])

    st.dataframe(
        customer_data[
            [
                "nr_proposta",
                "id_proposta",
                "dt_proposta",
                "dt_inicio_vigencia",
                "dt_fim_vigencia",
                "nm_municipio_propriedade",
                "sg_uf_propriedade",
                "evento_preponderante",
                "valor_indenização",
            ]
        ].T,
    )

extract_df = pd.read_csv(
    "C:\\Repos\\DeFraud\\data\\processed\\psr_sampled_featues.csv"
).query(f"nr_documento_segurado=='{selected}'")

with tab1:
    feature = st.selectbox(label="Select a feature", options=sorted(features))
    st.dataframe(extract_df[feature])

with tab2:
    st.header("Engineering")

with tab3:
    st.header("Checkpoint")
