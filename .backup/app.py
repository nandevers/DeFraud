import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.title("IsurEnv: An Insurance Environment")
st.markdown(
    """
    *This is an interactive tool to explore research results:*
    
    InsurEnv a insurance fraud detection environment to train Reinforcement \
    Learning algorithms.

    **Profile**
    In this section you can explore the raw dataset obtaind from the Brazilian Government website.

    **Features**
    The goal of this section is to allow the user to understand both the Feature Extraction process 
    as well as the Engeeniring that produces the training data. 
    
    """
)
