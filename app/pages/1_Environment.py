import streamlit as st
from PIL import Image

st.title('InsurEnv: A reinforcement learning Environment for Claim Fraud Detection')

image = Image.open('app/assets/env.png')

st.image(image, caption='Sunrise by the mountains')