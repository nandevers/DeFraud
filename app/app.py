import streamlit as st
import pandas as pd
import random


def main():
    st.title("Customer Profile")

    # Load the customer data
    df = pd.read_csv("../data/psr_sample.csv")

    # Display a dropdown to choose a customer
    selected_customer = st.selectbox(
        "Select a customer",
        df["nr_documento_segurado"],
        index=random.randint(0, df.shape[0] - 1),
    )

    # Filter the data for the selected customer
    customer_data = df[df["nr_documento_segurado"] == selected_customer]

    # Display the customer information
    st.write("**Customer Information**")
    st.write(
        customer_data[
            [
                "nm_razao_social",
                "cd_processo_susep",
                "nr_proposta",
                "id_proposta",
                "dt_proposta",
                "dt_inicio_vigencia",
                "dt_fim_vigencia",
                "nm_segurado",
                "nr_documento_segurado",
                "nm_municipio_propriedade",
                "sg_uf_propriedade",
            ]
        ]
    )

    # Calculate and display big numbers
    st.write("**Big Numbers**")
    st.write("---")
    st.write("Previous Proposals: ", customer_data["nr_proposta"].count())
    st.write("Previous Claims: ", customer_data["id_proposta"].nunique())
    st.write("Previous Settlements: ", customer_data["dt_fim_vigencia"].notnull().sum())


if __name__ == "__main__":
    main()
