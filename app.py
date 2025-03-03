import instructor
import joblib
import streamlit as st
import time
import re
import pandas as pd
import boto3
import os
import io
from dotenv import dotenv_values
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from langfuse import Langfuse
from langfuse.decorators import observe

#env = dotenv_values(".env")

# Konfiguracja dostępu do DigitalOcean Spaces
session = boto3.session.Session()
client = session.client(
    's3',
    region_name=os.getenv('AWS_REGION'),
    endpoint_url=os.getenv('AWS_ENDPOINT_URL_S3'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

BUCKET_NAME = 'biegacz'
MODEL_KEY = 'halfmarathon_model.pkl'

try:
    # Pobierz model z DigitalOcean Spaces bezpośrednio do pamięci
    response = client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    model_bytes = response['Body'].read()
    model_halfmarathon = joblib.load(io.BytesIO(model_bytes))
    print("Model został pomyślnie załadowany z DigitalOcean Spaces.")
except Exception as e:
    st.error(f"Wystąpił błąd podczas ładowania modelu z DigitalOcean Spaces: {e}")
    st.stop()  # Zatrzymaj aplikację, jeśli model nie może być załadowany

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

openai_client = OpenAI(api_key=st.session_state['openai_api_key'])
instructor_openai_client = instructor.from_openai(openai_client)

# Inicjalizacja Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),  # Opcjonalnie
)

# Definiowanie modelu danych przy użyciu Pydantic z wartościami domyślnymi
class UserData(BaseModel):
    Wiek: int | None = Field(default=None, ge=10, le=99)  # Wiek domyślnie None
    Płeć: str | None = Field(default=None)  # Płeć domyślnie None
    Czas_5_km: str = Field(default="00:00:00")  # Czas domyślnie "00:00:00"
    
def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']:
        return None
    if isinstance(time, str):
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", time):  # Sprawdzenie formatu HH:MM:SS
            return None
        try:
            time_parts = time.split(':')  # Zmieniono nazwę zmiennej na bardziej czytelną
            seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            if seconds < 600:  # Sprawdzenie, czy czas jest >= 10 minut
                return None
            return seconds
        except ValueError:
            raise ValueError("Niepoprawny format czasu. Czas powinien być w formacie HH:MM:SS.")  # Rzucamy wyjątek z komunikatem
    return None

@observe
def retrieve_structure(text):
    prompt = f"""
        Wyciagnij z tekstu następujące informacje:
        - Wiek (liczba całkowita, z zakresu 10<= x <= 99, w przeciwnym razie zwróć None)
        - Płeć (K dla kobiety, M dla mężczyzny, lub None, jeśli nie można ustalić)
        - Czas na 5 km (w formacie HH:MM:SS, lub 00:00:00 jeśli nie można ustalić)


        Postaraj się być jak najbardziej elastyczny i "inteligentny" w interpretacji danych.
        - Jeśli podano imię, spróbuj na jego podstawie określić płeć, zwróć również uwagę na czasowniki męskoosobowe i żeńskoosobowe.
        - Jeśli czas jest podany w innym formacie niż HH:MM:SS, spróbuj go przekonwertować.
        - Jeśli brakuje niektórych danych, oznacz je jako wartość domyślną.

        Tekst:
        '{text}'
    """

    res = instructor_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_model=UserData,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    dane = res.model_dump()

    # Sprawdzamy, czy wiek jest pustym stringiem (co oznacza, że nie został znaleziony lub jest niepoprawny)
    if dane["Wiek"] == "":
        dane["Wiek"] = None  # Ustawiamy wiek na None
    
    return dane

st.markdown("<h1 style='text-align: center; font-family: cursive;'>Przewidywany czas ukończenia półmaratonu wrocławskiego</h1>", unsafe_allow_html=True)

st.image("biegacze.png")  # Obrazek teraz umieszczony POD tytułem aplikacji!

if "dane_użytkownika" not in st.session_state:
    st.session_state["dane_użytkownika"] = ""

st.markdown(
    """
    <style>
    .custom-label {
        font-family: cursive !important; /* Zmieniamy czcionkę na bardziej "odręczną" */
        font-size: 16px; /* Możemy dostosować rozmiar czcionki */
        color: #888; /* Możemy dostosować kolor tekstu */
        text-align: center; /* Wyśrodkowujemy tekst */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='custom-label'>Proszę wpisz swoje dane: wiek, płeć oraz ile czasu zajmuje Ci pokonanie dystansu 5 km.</div>", unsafe_allow_html=True)

dane_użytkownika = st.text_area(
    label="",  # Dodajemy pusty label, aby uniknąć błędu
    value=st.session_state["dane_użytkownika"],
    height=100  # Dodajemy wysokość, aby pole tekstowe było większe
)

st.markdown(
    """
    <style>
    .stTextArea textarea {
        font-family: cursive; /* Zmieniamy czcionkę na bardziej "odręczną" */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("Sprawdź dane"):
    try:
        dane = retrieve_structure(dane_użytkownika)
        st.session_state["wiek"] = dane["Wiek"]
        st.session_state["plec"] = dane["Płeć"]
        st.session_state["czas_5km"] = dane["Czas_5_km"]

        # ... (usunięto wyświetlanie danych)

        dane_json = {
            "Wiek": st.session_state["wiek"],
            "Płeć": st.session_state["plec"],
            "5 km Czas": convert_time_to_seconds(st.session_state["czas_5km"])
        }

        brakujace_dane = []
        if dane_json["Wiek"] is None:
            brakujace_dane.append("wieku (przedział 10-99 lat)")
        if dane_json["Płeć"] is None:
            brakujace_dane.append("płci")
        if dane_json["5 km Czas"] is None:
            brakujace_dane.append("czasu na 5 km (format HH:MM:SS, minimum 10 minut)")

        if brakujace_dane:
            message_error = "Brakuje danych dla: " + ", ".join(brakujace_dane) + ". Proszę uzupełnij dane."
            st.error(message_error)
        else:
            success_placeholder = st.empty()  # Utworzenie placeholder-a dla komunikatu o sukcesie
            success_placeholder.success("Dane wprowadzone poprawnie. Rozpoczynam predykcję...")
            time.sleep(2)  # Wyświetlanie komunikatu przez 2 sekundy
            success_placeholder.empty() # Usunięcie komunikatu z placeholder-a

            df_predykcja = pd.DataFrame([dane_json])

            time.sleep(1)

            # Predykcja czasu półmaratonu
            try:
                predicted_time = model_halfmarathon.predict(df_predykcja)
                predicted_time_seconds = predicted_time[0] # Wyciągnij pojedynczą wartość

                # Konwersja sekund na format HH:MM:SS
                predicted_hours = int(predicted_time_seconds / 3600)
                predicted_minutes = int((predicted_time_seconds % 3600) / 60)
                predicted_seconds = int(predicted_time_seconds % 60)
                predicted_time_format = f"{predicted_hours:02d}:{predicted_minutes:02d}:{predicted_seconds:02d}"
                    
                title_placeholder = st.empty() # Utworzenie placeholder-a dla tytułu
                title_text = "Przewidywany czas ukończenia półmaratonu:" # Tekst tytułu

                full_title_text = "" # Inicjalizacja pustego stringa dla tytułu
                for char_title in title_text: # Iteracja po znakach w title_text
                    full_title_text += char_title # Dodawanie kolejnego znaku do stringa tytułu
                    title_placeholder.markdown(f"<div style='text-align: center; font-size: 24px; font-family: cursive;'>{full_title_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.1) # Małe opóźnienie, regulujące szybkość pojawiania się znaków tytułu (0.05 sekundy)

                time.sleep(1) # Pauza 1 sekunda przed wyświetleniem predicted_time_format - zmieniono na 1 sekundę

                predicted_time_placeholder = st.empty() # Utworzenie placeholder-a

                full_text = "" # Inicjalizacja pustego stringa
                for char in predicted_time_format: # Iteracja po znakach w predicted_time_format
                    full_text += char # Dodawanie kolejnego znaku do stringa
                    predicted_time_placeholder.markdown(f"<div style='text-align: center; font-size: 46px; font-weight: bold; font-family: cursive;'>{full_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.2) # Małe opóźnienie, regulujące szybkość pojawiania się znaków (0.05 sekundy)
                    
            except Exception as e:
                st.error(f"Wystąpił błąd podczas predykcji: {e}")
                
            time.sleep(1)
            if st.button("Wyczyść dane"): # Najprostszy przycisk - bez placeholder-a i time.sleep()
                st.session_state["dane_użytkownika"] = ""
                st.session_state["wiek"] = ""
                st.session_state["plec"] = ""
                st.session_state["czas_5km"] = ""
                    
                # WYRAŹNE ustawienie wartości text_area w session_state PRZED RERUN! - TO JEST KLUCZOWA ZMIANA!
                st.session_state["dane_użytkownika"] = ""

                st.error("---DEBUG: NAJPROSTSZY PRZYCISK 'Wyczyść dane' - KOD URUCHOMIONY! - st.stop() ---") # Bardzo wyraźny DEBUG
                st.stop()
                
    except ValueError as e:  # Dodajemy obsługę ValueError
        st.error(f"Błąd: {e}")  # Wyświetlamy komunikat o błędzie konwersji czasu

    except ValidationError as e:
        # ... (kod obsługi błędów bez zmian)
        errors = e.errors()
        missing_fields = []
        for error in errors:
            missing_fields.append(error['loc'][0])

        error_messages = []
        if 'Wiek' in missing_fields:
            error_messages.append("Brakuje wieku.")
        if 'Płeć' in missing_fields:
            error_messages.append("Brakuje płci.")
        if 'Czas_5_km' in missing_fields:
            error_messages.append("Brakuje czasu na 5 km.")

        st.error("Błąd: " + " ".join(error_messages))

    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd: {e}")
