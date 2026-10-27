import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# Valeurs réelles attendues par le modèle (en anglais)
YES_NO = ["Yes", "No"]
# Services regroupés en deux indices (doivent correspondre aux bornes de l'API :
# Security_Index de 0 à 4 et Streaming_Pack de 0 à 2)
SERVICES_SECURITE = [
    "Sécurité en ligne", "Sauvegarde en ligne",
    "Protection de l'équipement", "Assistance technique",
]
SERVICES_STREAMING = ["Streaming TV", "Streaming films"]
MAX_SECURITY_INDEX = len(SERVICES_SECURITE)
MAX_STREAMING_PACK = len(SERVICES_STREAMING)

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

with col2:
    st.subheader("Services")
    internet = st.selectbox("Service Internet", ["DSL", "Fiber optic", "No"], format_func=fmt)
    sans_internet = internet == "No"

    security_index = st.number_input(
        f"Services de sécurité souscrits (0 à {MAX_SECURITY_INDEX})",
        min_value=0, max_value=MAX_SECURITY_INDEX, value=0, step=1,
        disabled=sans_internet,
    )
    st.caption("Compte parmi : " + ", ".join(SERVICES_SECURITE))

    streaming_pack = st.number_input(
        f"Services de streaming souscrits (0 à {MAX_STREAMING_PACK})",
        min_value=0, max_value=MAX_STREAMING_PACK, value=0, step=1,
        disabled=sans_internet,
    )
    st.caption("Compte parmi : " + ", ".join(SERVICES_STREAMING))

if st.button("Prédire", type="primary"):
    # Le payload conserve les clés et valeurs exactes en anglais requises par l'API
    payload = {
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "InternetService": internet,
        # Un widget désactivé conserve sa valeur : on force 0 si le client n'a pas Internet
        "Security_Index": 0 if sans_internet else int(security_index),
        "Streaming_Pack": 0 if sans_internet else int(streaming_pack),
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()
        
        prob = result["churn_probability"]

        # Affichage sous forme de score de risque non calibré
        st.metric("Score de risque de churn", f"{prob * 100:.1f} / 100")
        st.progress(prob)

        st.caption(
            "Ce score indique le niveau de risque de résiliation "
            "estimé par le modèle. Il ne représente pas une "
            "probabilité réelle calibrée."
        )

        if result["churn"]:
            st.error("Risque élevé : le client risque de résilier.")
        else:
            st.success("Risque faible : le client devrait rester.")

    except requests.RequestException as e:
        st.error(f"Impossible de contacter l'API : {e}")