import streamlit as st
import pandas as pd
import json
import math

from lexcomp import Model

st.set_page_config(layout="wide")

st.sidebar.title("LexComp textual complexity tool")
page = st.sidebar.radio("Go to", ["Home", "Calculate complexity", "About", "Contact"])

mdl = Model()

if page == "Home":
    st.title("Welcome to the LexComp tool!")
    st.write("""
             This app provides metrics and insights about lexical and syntactic complexity 
             of texts.
    """)

elif page == "Calculate complexity":
    st.title("Assess text complexity")

    col1, col2 = st.columns([8, 4], gap='large')

    with col1:
        text = st.text_input(
            "Enter the text you want to examine or upload a file",
            placeholder=" Type your text"
        )

        if text:
            st.write(f"Got {mdl.predict_text(text)[0]}")

    with col2:
        st.write("Test text")

elif page == "About":
    st.write("Find something")