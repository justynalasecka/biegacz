import pandas as pd

from openai import OpenAI


load_dotenv()

model_halfmarathon = "halfmarathon_model.pkl"

@observe()
def get_info_langfuse_observed(user_input, model="gpt-4o"):
    prompt = """
    Wyciagnij z tekstu dane, zachowaj odpowieni format.
    "Płeć" przypisz dla "kobiety" wartość "K", dla "mężczyzny" wartość "M".
    "5 km Czas" odczytaj czas z zapisu np. 20:59.
    "Rocznik" oblicz odejmując od aktualnego roku (2025) wartość, którą użytkownik wpisze jako "wiek".
    Zwróć wartość jako obiekt JSON z następującymi kluczami:
    "5 km Czas" - as a string,
    "Rocznik" -  ma byc interpretowany jako liczba całkowita integer,
    "Płeć" - as string.
    Przykład finalny JSON:
    {
    "5 km Czas": "14:28",
    "Rocznik": "1997"
    "Płeć": "Kobieta"
    }
    """   
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role":"user",
            "content": user_input,
        },
    ]
    chat_completion = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages,
        model=model,
        name="dane_uzytkownika",
    )

    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except json.JSONDecodeError:
        output = {"Uwaga": "Nie udało się przeanalizować JSON. Odpowiedź: " + resp}
        return output

    required_fields = ["5 km Czas", "Rocznik", "Płeć"]
    missing_or_invalid_fields = [
        field for field in required_fields 
        if field not in output or not output[field] or (field == "Płeć" and output[field] == "Nieokreślono")
        ]

    if missing_or_invalid_fields:
        raise ValueError(f"Brakujące informacje: {', '.join(missing_or_invalid_fields)}")
