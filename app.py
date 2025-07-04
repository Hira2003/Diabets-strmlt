import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========================
# PAGE CONFIGURATION
# ========================

PAGES = {
    "Diabetes (Pima)": {
        "csv_path": "diabetes.csv",
        "target_col": "Outcome",
        "model_dir": "models1/",
        "model_files": {
            "Random Forest": "Random_Forest_after_clonal_selection.joblib",
            "Decision Tree": "Decision_Tree_after_clonal_selection.joblib",
            "SVM": "SVM_after_clonal_selection.joblib",
            "XGBoost": "XGBoost_after_clonal_selection.joblib",
            "MLP (Neural Network)": "MLP_Neural_Network_after_clonal_selection.joblib",
            "Logistic Regression": "Logistic_Regression_after_clonal_selection.joblib"
        },
        "categorical_map": None,
        "note": "Ce formulaire vous permet de saisir les paramètres médicaux d’un patient pour prédire le risque de diabète - Dataset Pima."
    },
    "Diabetes1 (Custom1)": {
        "csv_path": "diabetes1.csv",
        "target_col": "Outcome",
        "model_dir": "models2/",
        "model_files": {
            "Random Forest": "Random_Forest_after_selection.pkl",
            "Decision Tree": "Decision_Tree_after_selection.pkl",
            "SVM": "SVM_after_selection.pkl",
            "XGBoost": "XGBoost_after_selection.pkl",
            "MLP (Neural Network)": "MLP_(Neural_Network)_after_selection.pkl",
            "Logistic Regression": "Logistic_Regression_after_selection.pkl"
        },
        "categorical_map": None,
        "note": "Remplissez les champs pour prédire la présence de diabète selon les caractéristiques d'entrée (diabetes1.csv)."
    },
    "Diabetes2 (Custom2)": {
        "csv_path": "diabetes2.csv",
        "target_col": "class",
        "model_dir": "models3/",
        "model_files": {
            "Random Forest": "random_forest_clonal_selection_model.pkl",
            "Decision Tree": "decision_tree_clonal_selection_model.pkl",
            "SVM": "svm_clonal_selection_model.pkl",
            "XGBoost": "xgboost_clonal_selection_model.pkl",
            "MLP (Neural Network)": "mlp_(neural_network)_clonal_selection_model.pkl",
            "Logistic Regression": "logistic_regression_clonal_selection_model.pkl"
        },
        "categorical_map": {
            "Gender": ["Female", "Male"],
            "Polyuria": ["No", "Yes"],
            "Polydipsia": ["No", "Yes"],
            "sudden weight loss": ["No", "Yes"],
            "weakness": ["No", "Yes"],
            "Polyphagia": ["No", "Yes"],
            "Genital thrush": ["No", "Yes"],
            "visual blurring": ["No", "Yes"],
            "Itching": ["No", "Yes"],
            "Irritability": ["No", "Yes"],
            "delayed healing": ["No", "Yes"],
            "partial paresis": ["No", "Yes"],
            "muscle stiffness": ["No", "Yes"],
            "Alopecia": ["No", "Yes"],
            "Obesity": ["No", "Yes"]
        },
        "note": "Remplissez les champs pour prédire la présence de diabète selon les caractéristiques d'entrée (diabetes2.csv, features catégorielles incluses)."
    }
}

# ========================
# UTILS
# ========================

@st.cache_resource
def load_model(model_path):
    obj = joblib.load(model_path)
    # Try to extract model and feature_names if packed as dict
    if isinstance(obj, dict):
        if "model" in obj and "feature_names" in obj:
            return obj["model"], obj["feature_names"]
    # fallback: try to get feature_names_ if present
    if hasattr(obj, "feature_names_in_"):
        return obj, list(obj.feature_names_in_)
    return obj, None

@st.cache_data
def load_csv(csv_path):
    return pd.read_csv(csv_path)

def show_numeric_input(df, feature, value=None):
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())
    mean_value = float(df[feature].mean())
    try:
        if value is None or pd.isnull(value):
            value = mean_value
        else:
            value = float(value)
    except Exception:
        value = mean_value
    return st.number_input(
        f"{feature}", min_value=min_value, max_value=max_value, value=value
    )

def clean_numeric(val):
    try:
        if pd.isnull(val):
            return None
        if isinstance(val, str):
            val = val.strip()
        try:
            return int(float(val))
        except Exception:
            return float(val)
    except Exception:
        return None

# ========================
# STREAMLIT APP
# ========================

st.set_page_config(page_title="🩺 Prédiction du Diabète", layout="wide")
st.sidebar.title("Navigation")
page_choice = st.sidebar.radio("Choisissez la page :", list(PAGES.keys()))

config = PAGES[page_choice]
st.title(page_choice)
st.info(config["note"])

df = load_csv(config["csv_path"])
features = [col for col in df.columns if col != config["target_col"]]
categorical_map = config["categorical_map"]

# Model selection
model_files = config["model_files"]
model_name = st.selectbox("Choisissez le modèle de classification", list(model_files.keys()))
model_path = os.path.join(config["model_dir"], model_files[model_name])

try:
    model, model_features = load_model(model_path)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.header("Entrez les valeurs pour chaque caractéristique")

# --- Importer un exemple du dataset ---
example_row = None
example_idx = None

st.write("Choisissez une ligne du dataset à importer comme exemple (optionnel):")
row_options = df.index.tolist()
row_display = [f"Ligne {i+1}" for i in row_options]
selected_row = st.selectbox("Sélectionnez le numéro de ligne :", row_display, index=0)
example_button = st.button("Importer la ligne sélectionnée")

if example_button:
    example_idx = row_options[row_display.index(selected_row)]
    st.success(f"Exemple importé : ligne {example_idx+1} du dataset")
    example_row = df.iloc[example_idx]
else:
    example_row = None

user_input = {}

for feature in features:
    if categorical_map and feature in categorical_map:
        options = categorical_map[feature]
        if example_row is not None:
            val = clean_numeric(example_row[feature])
            try:
                selection = options[int(val)] if val is not None and 0 <= int(val) < len(options) else options[0]
            except Exception:
                selection = options[0]
        else:
            selection = options[0]
        selection = st.selectbox(f"{feature}", options, key=feature, index=options.index(selection))
        encoded = options.index(selection)
        user_input[feature] = encoded
    elif page_choice == "Diabetes2 (Custom2)" and feature == "Age":
        original_age_min = int(df["Age"].min())
        original_age_max = int(df["Age"].max())
        if example_row is not None:
            user_val = clean_numeric(example_row["Age"])
            if user_val is None:
                user_val = int(df["Age"].mean())
        else:
            user_val = int(df["Age"].mean())
        user_val = st.number_input(
            f"{feature} (original scale)",
            min_value=original_age_min, max_value=original_age_max,
            value=int(user_val),
            key=feature
        )
        user_input[feature] = (user_val - df["Age"].mean()) / df["Age"].std()
    else:
        if example_row is not None:
            value = clean_numeric(example_row[feature])
        else:
            value = None
        user_input[feature] = show_numeric_input(df, feature, value=value)

input_df = pd.DataFrame([user_input])

# Align input_df columns to model_features if available
if model_features is not None:
    # Add missing columns as 0, remove extra columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

if st.button("Prédire le diabète"):
    with st.spinner("Prédiction en cours..."):
        try:
            prediction = model.predict(input_df)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                st.success(f"Prédiction : {'Diabétique' if prediction == 1 else 'Non diabétique'}")
                st.info(f"Probabilité : {proba[1]*100:.2f}% diabétique, {proba[0]*100:.2f}% non diabétique")
            else:
                st.success(f"Prédiction : {'Diabétique' if prediction == 1 else 'Non diabétique'}")
        except Exception as ex:
            st.error(f"Erreur lors de la prédiction : {ex}")
            st.stop()

st.markdown("---")
st.caption("👩‍⚕️ Cette application utilise des modèles entraînés avec sélection de caractéristiques. Veuillez consulter un professionnel de santé pour toute décision médicale.")
