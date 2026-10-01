# 💳 Credit Card Fraud Detection System

![Run Model Tests](https://github.com/agarwalshubham-97/fraud-detection-system/actions/workflows/tests.yml/badge.svg)

An end-to-end machine learning project for detecting potentially fraudulent credit card transactions using a **Random Forest classifier**, **SHAP explainability**, and an interactive **Streamlit dashboard**.

The application supports single-transaction predictions, batch CSV predictions, configurable classification thresholds, model evaluation, confusion matrix analysis, ROC and Precision–Recall curves, threshold sensitivity analysis, feature importance, and transaction-level explanations.

---

## 🚀 Features

* 🤖 Random Forest fraud detection model
* 💳 Single transaction prediction
* 📂 Batch CSV transaction prediction
* 📊 Fraud probability prediction
* 🎯 Configurable classification threshold
* 🧮 Dynamic classification metrics
* 📈 ROC curve
* 📉 Precision–Recall curve
* 🧮 Confusion matrix
* 📊 Transaction prediction summary
* 🎚️ Threshold sensitivity analysis
* 🏆 Recommended threshold based on F1 Score
* ⬇️ Downloadable prediction results
* 🧠 SHAP model explanations
* 📊 Random Forest feature importance
* 🖥️ Interactive Streamlit dashboard
* 🧪 Automated unit and model tests
* ✅ 100% test coverage for `fraud_utils.py`

---

## 📊 Model Evaluation

Model performance is calculated using the real evaluation dataset:

```text
real_model_evaluation.csv
```

The evaluation dataset contains:

* `Actual` — true transaction class
* `Probability` — predicted fraud probability

The selected classification threshold converts probabilities into predicted classes:

```text
Probability ≥ Threshold → FRAUD
Probability < Threshold → NORMAL
```

The dashboard dynamically calculates:

| Metric    | Description                                                                     |
| --------- | ------------------------------------------------------------------------------- |
| Accuracy  | Overall prediction correctness                                                  |
| Precision | Percentage of predicted fraud transactions that are actually fraud              |
| Recall    | Percentage of actual fraud transactions correctly detected                      |
| F1 Score  | Balance between precision and recall                                            |
| ROC-AUC   | Ability of the model to distinguish fraud from normal transactions              |
| PR-AUC    | Precision–Recall performance, particularly useful for imbalanced classification |

Because fraud detection datasets are highly imbalanced, the project evaluates precision, recall, F1 Score, ROC-AUC, and PR-AUC in addition to accuracy.

---

## 🎯 Threshold Optimization

The dashboard allows the fraud classification threshold to be adjusted interactively.

It evaluates model performance across multiple threshold values and identifies the threshold with the highest F1 Score as the recommended threshold.

This demonstrates the trade-off between fraud detection and false positives:

```text
Lower Threshold
       ↓
More transactions classified as fraud
Higher Recall
Potentially more false positives

Higher Threshold
       ↓
Fewer transactions classified as fraud
Potentially higher Precision
Potentially more missed fraud
```

The threshold can be changed directly in the dashboard, allowing users to observe how classification metrics change.

---

## 🧠 Model Explainability

The project provides two complementary approaches to model explainability.

### 📊 Feature Importance

The Random Forest model's built-in feature importance scores provide a global view of which features contribute most to the model's decisions.

The dashboard displays:

* Feature importance visualization
* Top 10 most important features
* Feature importance scores

### 🔍 SHAP Explanations

SHAP (SHapley Additive exPlanations) is used to provide transaction-level explanations for individual predictions.

For each single-transaction prediction, the dashboard displays:

* Prediction result
* Fraud probability
* Risk level
* Top 10 feature contributions
* SHAP values
* Direction of contribution

The direction is displayed as:

```text
Increases fraud risk
Reduces fraud risk
No significant impact
```

This complements the global Random Forest feature importance analysis with an explanation specific to an individual prediction.

---

## 💳 Single Transaction Prediction

The dashboard allows users to select a complete transaction from the provided `test_transactions.csv` dataset.

The trained model expects 30 features:

```text
Time, V1–V28, Amount
```

For the selected transaction, the dashboard uses all 30 model features.

The prediction section displays:

* Predicted transaction class
* Fraud probability
* Risk level
* Current classification threshold
* Top feature contributions
* SHAP values
* Direction of feature influence

> **Note:** This interface uses complete sample transactions from the provided test dataset rather than collecting all 30 features manually from the user.

---

## 📂 Batch Prediction

Users can upload a CSV file containing transactions with the required model features.

The dashboard provides:

* Uploaded transaction data
* Predicted classes
* Fraud probabilities
* Total transaction count
* Normal transaction count
* Fraud transaction count
* Prediction summary
* Transaction summary
* Fraud vs. normal visualization
* Confusion matrix based on the saved real test-set evaluation data
* Downloadable prediction results

The expected model feature schema is:

```text
Time, V1, V2, ..., V28, Amount
```

---

## 📈 Real Model Evaluation

The dashboard evaluates model predictions using:

```text
real_model_evaluation.csv
```

It displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* ROC Curve
* Precision–Recall Curve
* Confusion Matrix
* Threshold sensitivity
* Recommended threshold

---

## 🧪 Testing

The project includes automated tests for the fraud detection utility functions and model-related functionality.

Run the test suite with:

```bash
python -m pytest --cov=fraud_utils --cov-report=term-missing --cov-fail-under=95 -q
```

Current test status:

```text
48 passed
100% coverage for fraud_utils.py
```

The project also includes a GitHub Actions workflow that automatically runs the test suite on pushes and pull requests targeting the `main` branch.

---

## 🛠️ Technologies Used

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* SHAP
* Matplotlib
* Seaborn
* Jupyter Notebook
* Pytest
* GitHub Actions

---

## 🧠 Machine Learning Workflow

```text
Credit Card Transaction Data
            ↓
Exploratory Data Analysis
            ↓
Data Preparation
            ↓
Train / Test Split
            ↓
Random Forest Training
            ↓
Model Evaluation
            ↓
ROC-AUC / PR-AUC / F1 Analysis
            ↓
Threshold Optimization
            ↓
Model Serialization
            ↓
SHAP Explainability
            ↓
Interactive Streamlit Dashboard
```

---

## 📁 Project Structure

```text
fraud-detection-system/
│
├── app.py
├── fraud_utils.py
├── README.md
├── requirements.txt
├── real_model_evaluation.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   ├── feature_names.pkl
│   └── model_config.pkl
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   └── 02_random_forest.ipynb
│
├── tests/
│   ├── test_fraud_utils.py
│   └── test_model.py
│
├── test_transactions.csv
└── test_mixed_transactions.csv
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/agarwalshubham-97/fraud-detection-system.git
```

### 2. Navigate to the project directory

```bash
cd fraud-detection-system
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will then be available locally in your browser.

---

## 🖥️ Dashboard Capabilities

### Model Performance

The dashboard provides interactive evaluation of classification performance across different fraud probability thresholds.

### Single Transaction Prediction

Select a complete transaction from the provided test dataset to generate a prediction and view its fraud probability, risk level, and SHAP explanation.

### Batch Prediction

Upload a CSV containing transaction records to generate predictions for multiple transactions.

### Threshold Analysis

Adjust the classification threshold and observe changes in:

* Accuracy
* Precision
* Recall
* F1 Score
* Fraud classifications

### Model Explainability

Explore both global feature importance and transaction-level SHAP explanations.

### Model Evaluation

Review:

* ROC curve
* Precision–Recall curve
* Confusion matrix
* Classification metrics
* Threshold sensitivity

---

## ⚠️ Limitations

This project is designed as a machine learning demonstration and portfolio project.

Important limitations include:

* The single-transaction interface uses complete sample transactions from the provided test dataset rather than collecting all 30 features manually.
* Model predictions depend on the quality and distribution of the training data.
* Threshold selection involves a trade-off between different classification metrics.
* This project is not intended to replace a production fraud detection system or financial risk-control process.

---

## 📌 Future Improvements

Potential future improvements include:

* Real-time transaction prediction
* Database integration
* REST API development
* Docker containerization
* Cloud deployment
* Automated model monitoring

---

## 👨‍💻 Author

**Shubham Kumar**

GitHub: https://github.com/agarwalshubham-97
