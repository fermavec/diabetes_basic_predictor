import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    modelo = joblib.load(BASE_DIR / "models" / "modelo_diabetes_rf.pkl")
    scaler = joblib.load(BASE_DIR / "models" / "escalador_diabetes.pkl")
    return modelo, scaler

modelo, scaler = load_artifacts()

FEATURE_NAMES = [
    "index", "year", "age",
    "race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other",
    "hypertension", "heart_disease",
    "bmi", "hbA1c_level", "blood_glucose_level",
    "gender_Male",
    "smoking_history_current", "smoking_history_ever", "smoking_history_former",
    "smoking_history_never", "smoking_history_not current",
    "location_Alaska", "location_Arizona", "location_Arkansas", "location_California",
    "location_Colorado", "location_Connecticut", "location_Delaware",
    "location_District of Columbia", "location_Florida", "location_Georgia",
    "location_Guam", "location_Hawaii", "location_Idaho", "location_Illinois",
    "location_Indiana", "location_Iowa", "location_Kansas", "location_Kentucky",
    "location_Louisiana", "location_Maine", "location_Maryland",
    "location_Massachusetts", "location_Michigan", "location_Minnesota",
    "location_Mississippi", "location_Missouri", "location_Montana",
    "location_Nebraska", "location_Nevada", "location_New Hampshire",
    "location_New Jersey", "location_New Mexico", "location_New York",
    "location_North Carolina", "location_North Dakota", "location_Ohio",
    "location_Oklahoma", "location_Oregon", "location_Pennsylvania",
    "location_Puerto Rico", "location_Rhode Island", "location_South Carolina",
    "location_South Dakota", "location_Tennessee", "location_Texas",
    "location_United States", "location_Utah", "location_Vermont",
    "location_Virgin Islands", "location_Virginia", "location_Washington",
    "location_West Virginia", "location_Wisconsin", "location_Wyoming",
    "comorbidity_index",
    "bmi_category_Normal", "bmi_category_Sobrepeso", "bmi_category_Obesidad",
    "glucose_category_Prediabetes", "glucose_category_Glucosa_Alta",
]

LOCATIONS = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Guam", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "United States", "Utah", "Vermont", "Virgin Islands", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
]


def build_features(age, bmi, hba1c, glucose, gender, smoking,
                   hypertension, heart_disease, race, location, year):
    if bmi < 18.5:
        bmi_cat = "Bajo_Peso"
    elif bmi < 25.0:
        bmi_cat = "Normal"
    elif bmi < 30.0:
        bmi_cat = "Sobrepeso"
    else:
        bmi_cat = "Obesidad"

    glucose_cat = "Glucosa_Normal" if glucose <= 140 else ("Prediabetes" if glucose <= 199 else "Glucosa_Alta")

    row = {name: 0 for name in FEATURE_NAMES}
    row["index"] = 0
    row["year"] = year
    row["age"] = float(age)
    row[f"race:{race}"] = 1
    row["hypertension"] = int(hypertension)
    row["heart_disease"] = int(heart_disease)
    row["bmi"] = float(bmi)
    row["hbA1c_level"] = float(hba1c)
    row["blood_glucose_level"] = int(glucose)
    row["gender_Male"] = 1 if gender == "Male" else 0
    if smoking != "No Info":
        row[f"smoking_history_{smoking}"] = 1
    if location != "Alabama":
        row[f"location_{location}"] = 1
    row["comorbidity_index"] = int(hypertension) + int(heart_disease)
    row["bmi_category_Normal"] = 1 if bmi_cat == "Normal" else 0
    row["bmi_category_Sobrepeso"] = 1 if bmi_cat == "Sobrepeso" else 0
    row["bmi_category_Obesidad"] = 1 if bmi_cat == "Obesidad" else 0
    row["glucose_category_Prediabetes"] = 1 if glucose_cat == "Prediabetes" else 0
    row["glucose_category_Glucosa_Alta"] = 1 if glucose_cat == "Glucosa_Alta" else 0

    return pd.DataFrame([row])[FEATURE_NAMES]


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Predictor de Diabetes", page_icon="🩺", layout="centered")
st.title("🩺 Predictor de Riesgo de Diabetes")
st.caption("Modelo Random Forest entrenado sobre 97,867 pacientes · Recall clase positiva: 90%")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Indicadores clínicos")
    age = st.slider("Edad", 2, 80, 45)
    bmi = st.slider("IMC (BMI)", 10.0, 95.0, 27.3, step=0.1)
    hba1c = st.slider("HbA1c (%)", 3.5, 9.0, 5.5, step=0.1)
    glucose = st.slider("Glucosa en sangre (mg/dL)", 80, 300, 140)

with col2:
    st.subheader("Datos del paciente")
    gender = st.radio("Género", ["Female", "Male"], horizontal=True)
    smoking = st.selectbox("Historial de tabaquismo",
                           ["No Info", "never", "former", "current", "not current", "ever"])
    hypertension = st.checkbox("Hipertensión")
    heart_disease = st.checkbox("Enfermedad cardíaca")

with st.expander("Datos demográficos adicionales"):
    race = st.selectbox("Raza", ["Other", "AfricanAmerican", "Asian", "Caucasian", "Hispanic"])
    location = st.selectbox("Estado (EE.UU.)", LOCATIONS)
    year = st.select_slider("Año del registro", options=list(range(2015, 2023)), value=2019)

st.divider()

if st.button("Predecir riesgo", type="primary", use_container_width=True):
    X = build_features(age, bmi, hba1c, glucose, gender, smoking,
                       hypertension, heart_disease, race, location, year)
    X_scaled = scaler.transform(X)
    pred = modelo.predict(X_scaled)[0]
    prob = modelo.predict_proba(X_scaled)[0][1]

    # ── Resultado ─────────────────────────────────────────────────────────────
    st.subheader("Resultado")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred == 1:
            st.error("**RIESGO ALTO**\n\nEl modelo detecta indicadores compatibles con diabetes.")
        else:
            st.success("**RIESGO BAJO**\n\nEl modelo no detecta indicadores de riesgo significativo.")

    with res_col2:
        st.metric("Probabilidad estimada de diabetes", f"{prob:.1%}")
        st.progress(float(prob))
        st.caption("Umbral de decisión: 0.50 · El modelo prioriza minimizar falsos negativos (recall 90%)")

    # ── Feature importance global ──────────────────────────────────────────────
    st.subheader("Factores con mayor peso en el modelo")
    importances = modelo.feature_importances_
    top_n = 12
    top_idx = np.argsort(importances)[-top_n:]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_vals = importances[top_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(top_names, top_vals, color="#4C8BF5")
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Top 12 variables del modelo (importancia global)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption("ⓘ Importancia global del bosque aleatorio — no es específica para este paciente.")
