import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/fraud_detection_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
config = joblib.load("models/model_config.pkl")

st.title("💳 Fraud Detection System")

st.write("Machine Learning based credit card fraud detection.")

st.success("Model loaded successfully!")

st.write("Number of features:", len(feature_names))
st.write("Detection threshold:", config["threshold"])


st.header("Transaction Prediction")

st.write("Enter transaction details to check for possible fraud.")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

time = st.number_input(
    "Transaction Time",
    min_value=0.0,
    value=0.0
)

if st.button("Check Transaction"):

    # Create transaction using the model's 30 required features
    transaction_data = {}

    for feature in feature_names:
        transaction_data[feature] = 0.0

    # Use the values entered by the user
    transaction_data["Time"] = time
    transaction_data["Amount"] = amount

    # Convert to DataFrame
    transaction = pd.DataFrame([transaction_data])

    # Make sure feature order is correct
    transaction = transaction[feature_names]

    # Get fraud probability
    probability = model.predict_proba(transaction)[0][1]

    # Apply saved threshold
    if probability >= config["threshold"]:
        result = "FRAUD"
    else:
        result = "NORMAL"

    st.subheader("Prediction Result")

    if result == "FRAUD":
        st.error("🚨 FRAUD TRANSACTION")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.write(f"Fraud Probability: **{probability * 100:.4f}%**")


st.header("📂 Batch Transaction Prediction")

st.write("Upload a CSV file containing transaction data.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Check required features
    missing_features = [
        feature for feature in feature_names
        if feature not in data.columns
    ]

    if missing_features:
        st.error("Missing required features:")
        st.write(missing_features)

    else:
        st.success("All required features are present!")

        # Keep only the features used by the model
        transaction_data = data[feature_names]

        probabilities = model.predict_proba(transaction_data)[:, 1]

        # Use the threshold selected during model evaluation
        threshold = config["threshold"]

        predictions = (probabilities >= threshold).astype(int)
        
        # Convert predictions into readable labels
        data["Prediction"] = [
            "FRAUD" if prediction == 1 else "NORMAL"
            for prediction in predictions
]

        # Convert probability to percentage
        data["Fraud Probability"] = probabilities * 100

        st.subheader("Prediction Results")

        st.dataframe(
            data[["Prediction", "Fraud Probability"]]
        )
        # Download prediction results
        csv_results = data.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_results,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )
        # Summary
        total_transactions = len(data)
        fraud_transactions = (predictions == 1).sum()
        normal_transactions = (predictions == 0).sum()

        st.subheader("📊 Transaction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
             st.metric("Total Transactions", total_transactions)

        with col2:
             st.metric("Normal Transactions", normal_transactions)

        with col3:
             st.metric("Fraud Transactions", fraud_transactions)

        st.write("Current threshold:", config["threshold"])
        st.subheader("📈 Fraud Probability by Transaction")

        chart_data = data[["Fraud Probability"]].copy()

        chart_data["Transaction"] = [
             f"Transaction {i}"
             for i in range(len(chart_data))
        ]

        chart_data = chart_data.set_index("Transaction")

        st.bar_chart(
             chart_data,
             y="Fraud Probability"
        )