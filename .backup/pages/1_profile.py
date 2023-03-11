import streamlit as st
import pandas as pd
import random


def profile():
    st.title("Customer Profile")

    # Load the customer data
    df = pd.read_csv("data/processed/psr_soja_pr_sample.csv")
    customer_options = df["nr_documento_segurado"]

    # Display a dropdown to choose a customer
    selected_customer = st.selectbox(
        label="Select a customer", options=customer_options, index=0
    )

    # Filter the data for the selected customer
    customer_data = df[df["nr_documento_segurado"] == selected_customer].sort_values(
        "nr_proposta"
    )

    # Display the customer information
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
        ]
    )

    # Calculate and display big numbers
    st.write("---")
    st.write("**Big Numbers**")
    st.write("Previous Proposals: ", customer_data["nr_proposta"].count())
    st.write("Previous Claims: ", customer_data["id_proposta"].nunique())
    st.write(
        "Previous Settlements: ", customer_data["valor_indenização"].notnull().sum()
    )


if __name__ == "__main__":
    profile()
