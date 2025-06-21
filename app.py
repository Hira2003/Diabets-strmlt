import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 📦 Model loading helpers
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def get_model_list():
    # Add or remove models here as needed
    models = {
        "Random Forest": "Random_Forest_after_clonal_selection.joblib",
        "Decision Tree": "Decision_Tree_after_clonal_selection.joblib",
        "SVM": "SVM_after_clonal_selection.joblib",
        "XGBoost": "XGBoost_after_clonal_selection.joblib",
        "MLP (Neural Network)": "MLP_Neural_Network_after_clonal_selection.joblib",
        "Logistic Regression": "Logistic_Regression_after_clonal_selection.joblib"
    }
    return models

# 📁 Paths (adjust if running outside Google Colab)
MODEL_DIR = "/content/drive/MyDrive/correction/"
CSV_PATH = "/content/drive/MyDrive/correction/diabetes.csv"

# 📌 Load data to get feature names
@st.cache_data
def load_csv():
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["Outcome"])
    return X

X = load_csv()
features = list(X.columns)

st.title("🔬 Diabetes Prediction App")

st.write("Ce formulaire vous permet de saisir les paramètres médicaux d’un patient pour prédire le risque de diabète à l’aide de modèles de machine learning.")

# 🚦 Model selection
model_files = get_model_list()
model_name = st.selectbox("Choisissez le modèle de classification", list(model_files.keys()))
model_path = MODEL_DIR + model_files[model_name]

# 🚀 Load selected model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# 🚧 User input
st.header("Entrez les valeurs pour chaque caractéristique")
user_input = {}
for feature in features:
    min_value = float(X[feature].min())
    max_value = float(X[feature].max())
    mean_value = float(X[feature].mean())
    user_input[feature] = st.number_input(
        f"{feature}", min_value=min_value, max_value=max_value, value=mean_value
    )

input_df = pd.DataFrame([user_input])

# 🔮 Predict
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
st.caption("👩‍⚕️ Cette application utilise des modèles entraînés après sélection de caractéristiques clonales. Veuillez consulter un professionnel de santé pour toute décision médicale.")