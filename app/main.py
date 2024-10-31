import streamlit as st
import pickle 
import pandas as pd



def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=":xray:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("This study aimed to predict breast cancer using different machine-learning approaches applying demographic, laboratory, and mammographic data.")

if __name__ == "__main__":
    main()
