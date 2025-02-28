import streamlit as st
import pandas as pd
import joblib
import json
import instructor
from pydantic import BaseModel
from dotenv import dotenv_values
from openai import OpenAI

env = dotenv_values(".env")

model_halfmarathon = joblib.load("halfmarathon_model.pkl")

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):#get albo przypuści skrypt albo wyrzuci none (nie wyrzuci błędu, tylko none)
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

openai_client = OpenAI(api_key=st.session_state['openai_api_key'])
instructor_openai_client = instructor.from_openai(openai_client)
#llm_client
   
st.title("Przewidywany czas ukończenia półmaratonu wrocławskiego")

if "dane_użytkownika" not in st.session_state:
    st.session_state["dane_użytkownika"] = ""

dane_użytkownika = st.text_area(
    label="Proszę wpisz swoje dane: wiek (lata całkowite), płeć (kobieta/mężczyzna) oraz ile czasu zajmuje Ci pokonanie dystansu 5 km (min:sek).",
    value=st.session_state["dane_użytkownika"]
)
