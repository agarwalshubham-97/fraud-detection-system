
import streamlit as st
import joblib
import pandas as pd
from fraud_utils import (
    apply_threshold,
    calculate_classification_metrics,
    validate_transaction_data,
)
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

model = joblib.load("models/fraud_detection_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
config = joblib.load("models/model_config.pkl")

# Load real model evaluation data
evaluation_data = pd.read_csv("real_model_evaluation.csv")
# Sidebar
st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("---")

st.sidebar.write("### Model Information")

st.sidebar.write(
    f"**Model:** Random Forest"
)

st.sidebar.write(
    f"**Features:** {len(feature_names)}"
)

st.sidebar.subheader("⚙️ Model Settings")

# Initialize threshold
if "threshold" not in st.session_state:
    st.session_state["threshold"] = float(config["threshold"])

evaluation_threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state["threshold"],
    step=0.01
)

# Keep session state updated
st.session_state["threshold"] = evaluation_threshold

st.sidebar.write(
    f"**Threshold:** {evaluation_threshold * 100:.0f}%"
)

st.sidebar.markdown("---")

st.sidebar.write("### Project")

st.sidebar.write(
    "Credit Card Fraud Detection System"
)

st.sidebar.write(
    "Built with Python, Scikit-learn and Streamlit."
)

st.title("💳 Credit Card Fraud Detection System")
st.caption(
    "Machine Learning powered credit card transaction "
    "risk analysis using Random Forest."
)

st.caption(
    "Machine Learning powered fraud detection dashboard"
)

st.write(
    "An ML-powered dashboard for detecting potentially fraudulent "
    "credit card transactions."
)

st.success("🟢 Random Forest model loaded successfully")
st.divider()

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.metric(
        "Model Features",
        len(feature_names)
    )

with info_col2:
    st.metric(
    "Detection Threshold",
    f"{evaluation_threshold * 100:.0f}%"
    )
st.divider()

st.header("📊 Model Performance")

# Real test-set data
y_actual = evaluation_data["Actual"]
y_probability = evaluation_data["Probability"]



# Calculate classification metrics
classification_metrics = calculate_classification_metrics(
    y_actual,
    y_probability,
    evaluation_threshold,
)

real_accuracy = classification_metrics["accuracy"]
real_precision = classification_metrics["precision"]
real_recall = classification_metrics["recall"]
real_f1 = classification_metrics["f1"]

# Calculate ranking metrics
real_roc_auc = roc_auc_score(
    y_actual,
    y_probability
)

real_pr_auc = average_precision_score(
    y_actual,
    y_probability
)

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric(
        "Accuracy",
        f"{real_accuracy * 100:.2f}%"
    )

    st.metric(
        "Precision",
        f"{real_precision * 100:.2f}%"
    )

with perf_col2:
    st.metric(
        "Recall",
        f"{real_recall * 100:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{real_f1 * 100:.2f}%"
    )

with perf_col3:
    st.metric(
        "ROC-AUC",
        f"{real_roc_auc * 100:.2f}%"
    )

    st.metric(
        "PR-AUC",
        f"{real_pr_auc * 100:.2f}%"
    )
st.subheader("📈 Threshold Performance Comparison")

threshold_values = sorted(
    set([
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        round(evaluation_threshold, 2)
    ])
)

threshold_results = []

for test_threshold in threshold_values:

    metrics = calculate_classification_metrics(
        y_actual,
        y_probability,
        test_threshold,
    )

    threshold_results.append({
        "Threshold": f"{test_threshold:.2f}",
        "Current Selection": (
            "✅ Current"
            if round(test_threshold, 2)
            == round(evaluation_threshold, 2)
            else ""
        ),
        "Accuracy (%)": round(
            metrics["accuracy"] * 100,
            2,
        ),
        "Precision (%)": round(
            metrics["precision"] * 100,
            2,
        ),
        "Recall (%)": round(
            metrics["recall"] * 100,
            2,
        ),
        "F1 Score (%)": round(
            metrics["f1"] * 100,
            2,
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

st.dataframe(
    threshold_df,
    width="stretch"
)
st.subheader("📈 Threshold Sensitivity")

chart_data = threshold_df.copy()

chart_data["Threshold"] = chart_data["Threshold"].astype(float)

chart_data = chart_data.set_index("Threshold")

st.line_chart(
    chart_data[
        [
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)"
        ]
    ]
)
st.subheader("🏆 Recommended Threshold")

best_threshold_row = threshold_df.loc[
    threshold_df["F1 Score (%)"].idxmax()
]

best_threshold = float(
    best_threshold_row["Threshold"]
)

st.success(
    f"Recommended threshold based on the highest F1 Score: "
    f"{best_threshold:.2f}"
)

best_col1, best_col2, best_col3 = st.columns(3)

with best_col1:
    st.metric(
        "Best F1 Score",
        f"{best_threshold_row['F1 Score (%)']:.2f}%"
    )

with best_col2:
    st.metric(
        "Precision at Best Threshold",
        f"{best_threshold_row['Precision (%)']:.2f}%"
    )

with best_col3:
    st.metric(
        "Recall at Best Threshold",
        f"{best_threshold_row['Recall (%)']:.2f}%"
    )
if round(evaluation_threshold, 2) == round(best_threshold, 2):

    st.success(
        "✅ The currently selected threshold matches "
        "the recommended threshold."
    )

else:

    threshold_difference = abs(
        evaluation_threshold - best_threshold
    )

    st.info(
        f"ℹ️ Current threshold: {evaluation_threshold:.2f} | "
        f"Recommended threshold: {best_threshold:.2f} | "
        f"Difference: {threshold_difference:.2f}"
    )
if st.button("🎯 Apply Recommended Threshold"):

    st.session_state["threshold"] = best_threshold

    st.rerun()
st.header("🔍 Single Transaction Prediction")

st.write(
    "Enter the transaction amount and time. "
    "The remaining model features are set to their "
    "baseline value for this demonstration."
)

st.info(
    "The trained model expects 30 features: Time, V1–V28, and Amount."
)

input_col1, input_col2 = st.columns(2)

with input_col1:
    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

with input_col2:
    time = st.number_input(
        "⏱️ Transaction Time",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

if st.button(
    "🔎 Check Transaction",
    type="primary",
    width="stretch"
):

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
    try:
        probability = model.predict_proba(
            transaction
        )[0][1]

    except Exception as error:
        st.error(
            f"Unable to generate prediction: {error}"
        )
        st.stop()

    # Apply saved threshold
    prediction = apply_threshold(
        [probability],
        evaluation_threshold,
    ).iloc[0]

    if prediction == 1:
        result = "FRAUD"
    else:
        result = "NORMAL"

    st.subheader("Prediction Result")

    probability_percent = probability * 100

    if result == "FRAUD":
        st.error("🚨 FRAUD TRANSACTION")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.metric(
        "Fraud Probability",
        f"{probability_percent:.2f}%"
    )

    if probability_percent < 10:
        st.success("🟢 Low Risk")

    elif probability_percent < 50:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")
st.divider()

st.header("📂 Batch Transaction Prediction")

st.write(
    "Upload a CSV file containing transaction data "
    "to analyze multiple transactions at once."
)

st.info(
    "Your CSV should contain all 30 model features: "
    "Time, V1–V28, and Amount."
)

uploaded_file = st.file_uploader(
    "📄 Choose a transaction CSV file",
    type=["csv"],
    help="Upload a CSV containing Time, V1–V28, and Amount."
)
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        st.error(
            "The uploaded CSV file is empty."
        )
        st.stop()

    except Exception as error:
        st.error(
            f"Unable to read the CSV file: {error}"
        )
        st.stop()

    if data.empty:
        st.error(
            "The uploaded CSV file contains no transaction data."
        )
        st.stop()

    st.subheader("📋 Uploaded Data")

    st.write(
        f"Showing the first 5 rows of {len(data):,} uploaded transactions."
    )

    st.dataframe(
        data.head(),
        width="stretch"
    )

    # Check required features
    missing_features = [
        feature for feature in feature_names
        if feature not in data.columns
    ]
    extra_columns = [
    column for column in data.columns
    if column not in feature_names
    ]

    if missing_features:
        st.error(
            "This CSV cannot be used for batch prediction."
        )

        st.warning(
            f"{len(missing_features)} required feature(s) "
            "are missing."
        )

        st.info(
            "Expected transaction features: "
            "Time, V1–V28, and Amount."
        )

        with st.expander("View detected columns in this CSV"):
            st.write(data.columns.tolist())

        with st.expander("View missing feature columns"):
            st.write(missing_features)

    else:
        st.success("All required features are present!")

        if extra_columns:
            st.warning(
                f"{len(extra_columns)} extra column(s) will "
                "not be used for prediction."
            )

            with st.expander("View extra columns"):
                st.write(extra_columns)

        try:
            transaction_data = validate_transaction_data(
                data,
                feature_names,
            )

        except ValueError as error:
            st.error(str(error))
            st.stop()

        try:
            probabilities = model.predict_proba(
                transaction_data
            )[:, 1]

        except Exception as error:
            st.error(
                f"Unable to generate predictions: {error}"
            )
            st.stop()

        # Use the threshold selected during model evaluation
        threshold = evaluation_threshold

        predictions = apply_threshold(
            probabilities,
            threshold,
        )
        fraud_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())
        total_transactions = len(predictions)
        fraud_percentage = (
            fraud_count / total_transactions * 100
            if total_transactions > 0
            else 0
        )

        normal_percentage = (
            normal_count / total_transactions * 100
            if total_transactions > 0
            else 0
        )
        st.info(
            f"Predictions are currently classified using a "
            f"threshold of {evaluation_threshold:.2f} "
            f"({evaluation_threshold * 100:.0f}%)."
        )

        st.subheader("📊 Prediction Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric(
                "Total Transactions",
                total_transactions
            )

        with summary_col2:
            st.metric(
                "Normal Transactions",
                normal_count,
                f"{normal_percentage:.2f}%"
            )

        with summary_col3:
            st.metric(
                "Fraud Transactions",
                fraud_count,
                f"{fraud_percentage:.2f}%"
            )

        
        # Convert predictions into readable labels
        data["Prediction"] = [
            "FRAUD" if prediction == 1 else "NORMAL"
            for prediction in predictions
        ]

        # Convert probability to percentage
        data["Fraud Probability"] = probabilities * 100

        st.subheader("🔎 Prediction Results")

        st.write(
            "Each transaction is classified as NORMAL or FRAUD "
            "using the saved model threshold."
        )

        st.dataframe(
            data[["Prediction", "Fraud Probability"]],
            width="stretch"
        ) 
        # Download prediction results
        csv_results = data.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_results,
            file_name="fraud_predictions.csv",
            mime="text/csv",
            width="stretch"
        )
        # Summary
        total_transactions = len(data)
        fraud_transactions = (predictions == 1).sum()
        normal_transactions = (predictions == 0).sum()
        fraud_rate = (fraud_transactions / total_transactions) * 100

        st.subheader("📊 Transaction Summary")

        st.write(
            "Overview of the transactions processed by the model."
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
             st.metric("Total Transactions", total_transactions)

        with col2:
             st.metric("Normal Transactions", normal_transactions)

        with col3:
             st.metric("Fraud Transactions", fraud_transactions)

        with col4:
             st.metric(
                 "Fraud Rate",
                 f"{fraud_rate:.2f}%"
            )
        st.write(
            "Current threshold:",
            f"{evaluation_threshold:.2f}"
        )

        # Fraud vs Normal chart
        st.subheader("🚨 Fraud vs Normal Transactions")

        st.write(
            "Comparison of normal and potentially fraudulent "
            "transactions detected by the model."
        )

        status_chart = pd.DataFrame({
            "Status": ["Normal", "Fraud"],
            "Count": [normal_transactions, fraud_transactions]
        })

        st.bar_chart(
            status_chart.set_index("Status"),
            width="stretch"
        )

        # Fraud probability chart
        

        st.write(
            "Fraud probability assigned to each uploaded transaction."
        )

        
        chart_data = data[["Fraud Probability"]].copy()
        chart_data["Transaction"] = [
            f"Transaction {i + 1}"
            for i in range(len(chart_data))
        ]

        chart_data = chart_data.set_index("Transaction")

        st.bar_chart(
            chart_data,
            y="Fraud Probability",
            
        )
        
        
        st.divider()
        st.subheader("🧮 Confusion Matrix")

        y_pred = (
            y_probability >= evaluation_threshold
        ).astype(int)
        cm = confusion_matrix(
            y_actual,
            y_pred
        )

        tn, fp, fn, tp = cm.ravel()

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Fraud"],
            columns=["Predicted Normal", "Predicted Fraud"]
        )

        st.dataframe(
            cm_df,
            width="stretch"
        )

        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

        with cm_col1:
            st.metric("True Negative", tn)

        with cm_col2:
            st.metric("False Positive", fp)

        with cm_col3:
            st.metric("False Negative", fn)

        with cm_col4:
            st.metric("True Positive", tp)
        
 # Display confusion matrix
st.divider()

st.header("📈 Real Model Evaluation")
# Real test-set data
y_actual = evaluation_data["Actual"]
y_probability = evaluation_data["Probability"]



# ROC Curve
roc_fpr, roc_tpr, _ = roc_curve(
    y_actual,
    y_probability
)

roc_data = pd.DataFrame({
    "False Positive Rate": roc_fpr,
    "True Positive Rate": roc_tpr
})

st.subheader("📊 ROC Curve")

st.line_chart(
    roc_data.set_index("False Positive Rate")
)



roc_col1, roc_col2 = st.columns(2)

with roc_col1:
    st.metric(
        "ROC-AUC",
        f"{real_roc_auc:.4f}"
    )

with roc_col2:
    st.metric(
        "PR-AUC",
        f"{real_pr_auc:.4f}"
    )    

# Precision-Recall Curve
pr_precision, pr_recall, _ = precision_recall_curve(
    y_actual,
    y_probability
)

pr_data = pd.DataFrame({
    "Recall": pr_recall,
    "Precision": pr_precision
})

st.subheader("📉 Precision–Recall Curve")

st.line_chart(
    pr_data.set_index("Recall"))

st.write(
    f"**PR-AUC:** {real_pr_auc:.4f}"
)
# ============================================================
# Feature Importance
# ============================================================
st.header("🧠 Model Explainability")

st.subheader("📊 Feature Importance")

if hasattr(model, "feature_importances_"):

    feature_importance_data = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance_data = feature_importance_data.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        feature_importance_data.set_index("Feature")
    )

    st.subheader("🔝 Top 10 Most Important Features")

    st.dataframe(
        feature_importance_data.head(10),
        width="stretch"
    )

else:

    st.info(
        "Feature importance is not available for this model."
    )