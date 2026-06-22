import sys
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "modelo_diabetes_rf.pkl"
SCALER_PATH = BASE_DIR / "models" / "escalador_diabetes.pkl"

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


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.drop_duplicates()
    df.reset_index(inplace=True)

    df = df[df["age"] >= 2].copy()
    df = df[df["gender"] != "Other"].copy()
    df.reset_index(drop=True, inplace=True)

    df = pd.get_dummies(df, columns=["gender", "smoking_history", "location"], drop_first=True)

    bins_bmi = [0, 18.5, 24.9, 29.9, 100]
    labels_bmi = ["Bajo_Peso", "Normal", "Sobrepeso", "Obesidad"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bins_bmi, labels=labels_bmi)

    bins_glucose = [0, 140, 199, 500]
    labels_glucose = ["Glucosa_Normal", "Prediabetes", "Glucosa_Alta"]
    df["glucose_category"] = pd.cut(df["blood_glucose_level"], bins=bins_glucose, labels=labels_glucose)

    df["comorbidity_index"] = df["hypertension"] + df["heart_disease"]

    df = pd.get_dummies(df, columns=["bmi_category", "glucose_category"], drop_first=True)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    if "diabetes" in df.columns:
        df = df.drop("diabetes", axis=1)

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_NAMES]


def predict(input_csv: str) -> None:
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_raw = pd.read_csv(input_csv, sep=",")
    df = preprocess(df_raw)
    X_scaled = scaler.transform(df)
    predictions = modelo.predict(X_scaled)
    probabilities = modelo.predict_proba(X_scaled)[:, 1]

    result_df = df_raw[(df_raw["age"] >= 2) & (df_raw["gender"] != "Other")].copy()
    result_df = result_df.reset_index(drop=True)
    result_df["prediccion_diabetes"] = predictions
    result_df["probabilidad_diabetes"] = probabilities.round(3)

    output_path = input_csv.replace(".csv", "_predicciones.csv")
    result_df.to_csv(output_path, index=False)

    total = len(predictions)
    positivos = int(predictions.sum())
    print(f"Pacientes analizados: {total}")
    print(f"Diagnostico positivo (diabetes): {positivos} ({positivos / total * 100:.1f}%)")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py <ruta_archivo_csv>")
        print("Ejemplo: python predict.py data/nuevos_pacientes.csv")
        sys.exit(1)
    predict(sys.argv[1])
