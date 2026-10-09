import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# Valeurs réelles attendues par le modèle (en anglais)
YES_NO = ["Yes", "No"]
INTERNET_EXTRA = ["No", "Yes", "No internet service"]

# Dictionnaire de traduction pour l'interface visuelle
TRADUCTION = {
    "Yes": "Oui",
    "No": "Non",
    "Female": "Femme",
    "Male": "Homme",
    "No internet service": "Pas de service Internet",
    "No phone service": "Pas de service téléphonique",
    "DSL": "DSL",
    "Fiber optic": "Fibre optique",
    "Month-to-month": "Mensuel",
    "One year": "Un an",
    "Two year": "Deux ans",
    "Electronic check": "Chèque électronique",
    "Mailed check": "Chèque par courrier",
    "Bank transfer (automatic)": "Virement bancaire (automatique)",
    "Credit card (automatic)": "Carte de crédit (automatique)",
}

# Fonction auxiliaire pour traduire les options à l'écran
def fmt(option):
    return TRADUCTION.get(option, option)

st.title("Prédiction du Churn Client")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Client")
    gender = st.selectbox("Sexe", ["Female", "Male"], format_func=fmt)
    senior = st.selectbox("Senior / Personne âgée", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    partner = st.selectbox("En couple", YES_NO, format_func=fmt)
    dependents = st.selectbox("Personnes à charge", YES_NO, format_func=fmt)
    tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    st.subheader("Contrat et facturation")
    contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"], format_func=fmt)
    paperless = st.selectbox("Facturation sans papier", YES_NO, format_func=fmt)
    payment = st.selectbox(
        "Moyen de paiement",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        format_func=fmt,
    )
    monthly = st.number_input("Frais mensuels (€)", 0.0, 200.0, 70.0, step=0.5)
    total = st.number_input("Frais totaux (€)", 0.0, 10000.0, float(tenure * monthly), step=1.0)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Service téléphonique", YES_NO, format_func=fmt)
    multiple = st.selectbox("Lignes multiples", ["No", "Yes", "No phone service"], format_func=fmt)
    internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"], format_func=fmt)
    online_sec = st.selectbox("Sécurité en ligne", INTERNET_EXTRA, format_func=fmt)
    online_bkp = st.selectbox("Sauvegarde en ligne", INTERNET_EXTRA, format_func=fmt)
    device = st.selectbox("Protection de l'équipement", INTERNET_EXTRA, format_func=fmt)
    tech = st.selectbox("Assistance technique", INTERNET_EXTRA, format_func=fmt)
    stream_tv = st.selectbox("Streaming TV", INTERNET_EXTRA, format_func=fmt)
    stream_mov = st.selectbox("Streaming films", INTERNET_EXTRA, format_func=fmt)

if st.button("Prédire", type="primary"):
    # Le payload conserve les clés et valeurs exactes en anglais requises par l'API
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bkp,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        prob = result["churn_probability"]
        st.metric("Probabilité de churn", f"{prob:.1%}")
        st.progress(prob)
        if result["churn"]:
            st.error("Risque élevé : le client risque de résilier.")
        else:
            st.success("Risque faible : le client devrait rester.")
    except requests.RequestException as e:
        st.error(f"Impossible de contacter l'API : {e}")