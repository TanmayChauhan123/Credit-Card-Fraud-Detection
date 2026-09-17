import streamlit as st
import joblib
import numpy as np
import json

# Load trained model and threshold
model = joblib.load("fraud_detection_model.pkl")
threshold = joblib.load("fraud_threshold.pkl")

# Load demo transactions
with open("demo_transactions.json", "r") as file:
    demo_transactions = json.load(file)


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

input_mode = st.radio(
    "Choose Input Mode",
    ["Manual Transaction", "Demo Transaction"],
    horizontal=True
)

if input_mode == "Manual Transaction":

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

    with st.expander(
        "Advanced Transaction Features (V1–V28)",
        expanded=False
    ):

        v_features = {}

        columns = st.columns(4)

        for i in range(1, 29):

            column = columns[(i - 1) % 4]

            with column:
                v_features[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0
                )


else:

    st.subheader("Demo Transaction")

    demo_type = st.selectbox(
        "Choose Demo Transaction",
        ["Legitimate Example", "Fraud Example"]
    )

    st.info(
        "A representative transaction from the test set will be used."
    )


# Create buttons for prediction and reset

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


# Reset inputs
if reset_transaction:
    st.rerun()


# Make prediction
if check_transaction:

    # Create input data based on the selected input mode
    if input_mode == "Manual Transaction":

        input_data = [time]

        for i in range(1, 29):
            input_data.append(v_features[f"V{i}"])

        input_data.append(amount)

    else:

        feature_order = (
            ["Time"] +
            [f"V{i}" for i in range(1, 29)] +
            ["Amount"]
        )

        input_data = [
            demo_transactions[demo_type][feature]
            for feature in feature_order
        ]

    # Convert input into a 2D array
    input_data = np.array(input_data).reshape(1, -1)

    # Get fraud probability
    fraud_probability = model.predict_proba(input_data)[0][1]

    # Apply tuned threshold
    prediction = int(fraud_probability >= threshold)

    # Display prediction result
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    with result_col2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.2f}"
        )

    st.progress(float(fraud_probability))

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


# Application information
st.divider()

st.caption(
    "Model: Random Forest Classifier | "
    "Threshold: 0.30 | "
    "Built with Python, Scikit-learn and Streamlit"
)