# Predicción de Diabetes con Machine Learning

Proyecto de clasificación binaria para predecir el riesgo de diabetes en pacientes utilizando un modelo Random Forest optimizado.

## Estructura del proyecto

```
predict_diabetes/
├── notebooks/
│   └── EjercicioPracticoDiabetes.ipynb   # Pipeline completo: EDA, entrenamiento y evaluación
├── models/
│   ├── modelo_diabetes_rf.pkl             # Random Forest Classifier (200 árboles, optimizado)
│   └── escalador_diabetes.pkl             # StandardScaler ajustado en datos de entrenamiento
├── data/                                  # No incluido en el repositorio (ver sección Dataset)
├── app.py                                 # Tablero interactivo con Streamlit
├── predict.py                             # Script de inferencia por lotes (CSV)
├── requirements.txt
└── .gitignore
```

## Dataset

El dataset **no está incluido en este repositorio** por su tamaño. Se trata de un conjunto de 100,000 registros de pacientes de EE.UU. con variables clínicas y demográficas. Distribución: 91.5% sin diabetes / 8.5% con diabetes.

**Variables:** `year`, `gender`, `age`, `location`, `race:AfricanAmerican`, `race:Asian`, `race:Caucasian`, `race:Hispanic`, `race:Other`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `hbA1c_level`, `blood_glucose_level`, `diabetes`.

## Pipeline de entrenamiento

El notebook documenta las siguientes etapas:

1. **EDA** — Exploración y comprensión del dataset
2. **Limpieza** — Eliminación de duplicados, filtros de edad y género
3. **Visualización** — Distribuciones, boxplots y mapa de correlación
4. **Preprocesamiento** — One-hot encoding, normalización (StandardScaler)
5. **Balanceo** — Comparación entre SMOTE y RandomUnderSampler
6. **Feature Engineering** — Categorías clínicas de BMI y glucosa, índice de comorbilidad
7. **Modelado** — Baseline (Regresión Logística) y comparación de algoritmos
8. **Optimización** — GridSearchCV con validación cruzada (cv=5)
9. **Exportación** — Serialización del modelo y scaler con joblib

## Modelo final

| Métrica       | Valor |
|---------------|-------|
| Accuracy      | 90%   |
| Recall        | 90%   |
| Precisión     | 47%   |
| F1-Score      | 0.62  |

El modelo prioriza el **recall** para minimizar falsos negativos (pacientes enfermos sin diagnosticar), lo que lo hace adecuado como herramienta de tamizaje médico.

## Demo interactiva (Streamlit)

```bash
# Activar entorno virtual (obligatorio — todas las dependencias están aisladas aquí)
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Lanzar la app
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`. Permite ingresar datos de un paciente con sliders y selectores, y devuelve la predicción con probabilidad y un gráfico de importancia de variables.

## Script de inferencia por lotes

```bash
python predict.py data/nuevos_pacientes.csv
```

El archivo de entrada debe incluir las mismas columnas del dataset original (ver sección Dataset). Los resultados se guardan como `nuevos_pacientes_predicciones.csv` con dos columnas adicionales:
- `prediccion_diabetes`: 0 (negativo) o 1 (positivo)
- `probabilidad_diabetes`: probabilidad estimada (0.0 - 1.0)

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** todas las dependencias se instalan dentro de `venv/` y no afectan el entorno global del sistema.

## Tecnologías

- Python 3.x
- scikit-learn 1.9.0
- imbalanced-learn 0.14.2
- pandas 3.0.3
- matplotlib / seaborn
- Streamlit 1.58.0
