# Diabetes Risk Prediction with Machine Learning

Binary classification project to predict diabetes risk in patients using an optimized Random Forest model.

## Project Structure

```
predict_diabetes/
├── notebooks/
│   └── EjercicioPracticoDiabetes.ipynb   # Full pipeline: EDA, training and evaluation
├── models/
│   ├── modelo_diabetes_rf.pkl             # Random Forest Classifier (200 trees, optimized)
│   └── escalador_diabetes.pkl             # StandardScaler fitted on training data
├── data/                                  # Not included in the repository (see Dataset section)
├── app.py                                 # Interactive Streamlit dashboard
├── predict.py                             # Batch inference script (CSV input)
├── requirements.txt
└── .gitignore
```

## Dataset

The dataset is **not included in this repository** due to its size. It consists of 100,000 patient records from the United States with clinical and demographic variables. Class distribution: 91.5% non-diabetic / 8.5% diabetic.

**Features:** `year`, `gender`, `age`, `location`, `race:AfricanAmerican`, `race:Asian`, `race:Caucasian`, `race:Hispanic`, `race:Other`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `hbA1c_level`, `blood_glucose_level`, `diabetes`.

## Training Pipeline

The notebook documents the following stages:

1. **EDA** — Dataset exploration and understanding
2. **Cleaning** — Duplicate removal, age and gender filters
3. **Visualization** — Distributions, boxplots and correlation heatmap
4. **Preprocessing** — One-hot encoding, normalization (StandardScaler)
5. **Balancing** — SMOTE vs. RandomUnderSampler comparison
6. **Feature Engineering** — Clinical BMI and glucose categories, comorbidity index
7. **Modeling** — Logistic Regression baseline and algorithm comparison
8. **Optimization** — GridSearchCV with cross-validation (cv=5)
9. **Export** — Model and scaler serialization with joblib

## Final Model

| Metric    | Value |
|-----------|-------|
| Accuracy  | 90%   |
| Recall    | 90%   |
| Precision | 47%   |
| F1-Score  | 0.62  |

The model prioritizes **recall** to minimize false negatives (sick patients going undiagnosed), making it suitable as a medical screening tool.

## Interactive Demo (Streamlit)

```bash
# Activate virtual environment (required — all dependencies are isolated here)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Launch the app
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. Enter patient data using sliders and dropdowns to get a prediction with probability score and a feature importance chart.

## Batch Inference Script

```bash
python predict.py data/new_patients.csv
```

The input file must include the same columns as the original dataset (see Dataset section). Results are saved as `new_patients_predicciones.csv` with two additional columns:
- `prediccion_diabetes`: 0 (negative) or 1 (positive)
- `probabilidad_diabetes`: estimated probability (0.0 - 1.0)

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** all dependencies are installed inside `venv/` and do not affect the global system environment.

## Tech Stack

- Python 3.x
- scikit-learn 1.9.0
- imbalanced-learn 0.14.2
- pandas 3.0.3
- matplotlib / seaborn
- Streamlit 1.58.0
