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
            "MLP (Neural Network)": "MLP_Neural_Network_after_selection.pkl",
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
            "Random Forest": "Random_Forest_after_selection.pkl",
            "Decision Tree": "Decision_Tree_after_selection.pkl",
            "SVM": "SVM_after_selection.pkl",
            "XGBoost": "XGBoost_after_selection.pkl",
            "MLP (Neural Network)": "MLP_Neural_Network_after_selection.pkl",
            "Logistic Regression": "Logistic_Regression_after_selection.pkl"
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
    return joblib.load(model_path)

@st.cache_data
def load_csv(csv_path):
    return pd.read_csv(csv_path)

def show_numeric_input(df, feature):
    min_value = float(df[feature].min())
    max_value = float(df[feature].max())
    mean_value = float(df[feature].mean())
    return st.number_input(
        f"{feature}", min_value=min_value, max_value=max_value, value=mean_value
    )

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
    model = load_model(model_path)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.header("Entrez les valeurs pour chaque caractéristique")

user_input = {}
for feature in features:
    if categorical_map and feature in categorical_map:
        options = categorical_map[feature]
        selection = st.selectbox(f"{feature}", options, key=feature)
        encoded = options.index(selection)
        user_input[feature] = encoded
    elif page_choice == "Diabetes2 (Custom2)" and feature == "Age":
        original_age_min = int(df["Age"].min())
        original_age_max = int(df["Age"].max())
        user_val = st.number_input(
            f"{feature} (original scale)",
            min_value=original_age_min, max_value=original_age_max,
            value=int(df["Age"].mean()),
            key=feature
        )
        user_input[feature] = (user_val - df["Age"].mean()) / df["Age"].std()
    else:
        user_input[feature] = show_numeric_input(df, feature)

input_df = pd.DataFrame([user_input])

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
