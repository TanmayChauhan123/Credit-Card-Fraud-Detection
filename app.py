import streamlit as st
import joblib
import numpy as np


# Load trained model and threshold
model = joblib.load("fraud_detection_model.pkl")
threshold = joblib.load("fraud_threshold.pkl")


# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# Title
st.title("💳 Credit Card Fraud Detection")

st.write(
    "This application uses a trained Random Forest model "
    "to predict whether a credit card transaction is potentially fraudulent."
)

st.info(f"Classification threshold: {threshold}")

# Transaction input section
st.subheader("Enter Transaction Details")

st.write(
    "Enter the transaction values below to check whether "
    "the transaction is potentially fraudulent."
)

time = st.number_input(
    "Time",
    min_value=0.0,
    value=0.0
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=0.0
)

with st.expander("Advanced Transaction Features (V1–V28)", expanded=False):

    v_features = {}

    columns = st.columns(4)

    for i in range(1, 29):
        column = columns[(i - 1) % 4]

        with column:
            v_features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0
            )

# Create a button for prediction
col1, col2 = st.columns(2)

with col1:
    check_transaction = st.button(
        "🔍 Check Transaction",
        type="primary",
        use_container_width=True
    )

with col2:
    reset_transaction = st.button(
        "🔄 Reset Inputs",
        use_container_width=True
    )

if reset_transaction:
    st.rerun()

if check_transaction:

    # Create input data in the same order as the training features
    input_data = [time]

    for i in range(1, 29):
        input_data.append(v_features[f"V{i}"])

    input_data.append(amount)

    # Convert input into a 2D array
    input_data = np.array(input_data).reshape(1, -1)

    # Get fraud probability
    fraud_probability = model.predict_proba(input_data)[0][1]

    # Apply tuned threshold
    prediction = int(fraud_probability >= threshold)

    # Display prediction result
    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability:.2%}"
    )

    st.progress(float(fraud_probability))

    st.write(f"Classification threshold: **{threshold:.2f}**")

    if prediction == 1:
        st.error("⚠️ Potentially Fraudulent Transaction")
        st.write(
            "The predicted fraud probability is above the selected "
            "classification threshold."
        )
    else:
        st.success("✅ Legitimate Transaction")
        st.write(
            "The predicted fraud probability is below the selected "
            "classification threshold."
        )
