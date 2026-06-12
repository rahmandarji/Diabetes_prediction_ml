import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("Diabetes Risk Prediction")
st.write(
    "Enter patient information below. The app uses the saved XGBoost model, scaler, "
    "feature names, and threshold from training."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, feature_names, threshold


model, scaler, feature_names, threshold = load_artifacts()

NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
GENDER_OPTIONS = ["Female", "Male"]
SMOKING_OPTIONS = [
    "No Info",
    "current",
    "ever",
    "former",
    "never",
    "not current",
]

st.header("Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        hbA1c = st.number_input("HbA1c level", min_value=0.0, max_value=20.0, value=5.6, step=0.1)

    with col2:
        glucose = st.number_input(
            "Blood glucose level",
            min_value=0.0,
            max_value=600.0,
            value=120.0,
            step=1.0,
        )
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", GENDER_OPTIONS)
    with col4:
        smoking_history = st.selectbox("Smoking history", SMOKING_OPTIONS)

    submitted = st.form_submit_button("Predict diabetes risk")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "bmi": bmi,
                "HbA1c_level": hbA1c,
                "blood_glucose_level": glucose,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "gender": gender,
                "smoking_history": smoking_history,
            }
        ]
    )

    encoded_df = pd.get_dummies(
        input_df,
        columns=["gender", "smoking_history"],
        drop_first=True,
    )

    for col in encoded_df.select_dtypes(include="bool").columns:
        encoded_df[col] = encoded_df[col].astype(int)

    encoded_df = encoded_df.reindex(columns=feature_names, fill_value=0)

    encoded_df[NUMERICAL_FEATURES] = scaler.transform(encoded_df[NUMERICAL_FEATURES])

    probability = float(model.predict_proba(encoded_df)[:, 1][0])
    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High diabetes risk detected")
    else:
        st.success("Low diabetes risk detected")

    st.write(f"Probability of diabetes: **{probability:.3f}**")
    st.write(f"Decision threshold: **{threshold:.2f}**")

    st.caption(
        "The scaler is required because the XGBoost model was trained on scaled numerical features. "
        "The same scaler must be applied during inference so the input matches the training distribution."
    )

st.markdown("---")
st.markdown("### Model files required")
st.code(
    "diabetes_model.pkl\nscaler.pkl\nfeature_names.pkl\nthreshold.pkl",
    language="text",
)
