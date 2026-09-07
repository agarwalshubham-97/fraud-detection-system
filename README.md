# 💳 Credit Card Fraud Detection System

![Run Model Tests](https://github.com/agarwalshubham-97/fraud-detection-system/actions/workflows/tests.yml/badge.svg)

An end-to-end machine learning project for detecting potentially fraudulent credit card transactions using a **Random Forest classifier** and an interactive **Streamlit dashboard**.

The application supports individual transaction predictions, batch CSV predictions, configurable classification thresholds, real test-set evaluation, confusion matrix analysis, ROC and Precision–Recall curves, and threshold sensitivity analysis.

---

## 🚀 Features

- 🤖 Random Forest fraud detection model
- 💳 Single transaction prediction
- 📂 Batch CSV transaction prediction
- 📊 Fraud probability prediction
- 🎯 Configurable classification threshold
- 🧮 Dynamic classification metrics
- 📉 Precision–Recall curve
- 📈 ROC curve
- 🧮 Confusion matrix
- 📊 Transaction prediction summary
- 🎚️ Threshold sensitivity analysis
- 🏆 Recommended threshold based on F1 Score
- ⬇️ Downloadable prediction results
- 🧠 SHAP model explanations
- 📊 Feature importance visualization
- 🖥️ Interactive Streamlit dashboard

---

## 📊 Model Evaluation

Model performance is calculated using the real evaluation dataset:

```text
real_model_evaluation.csv
```
The application uses:

- `Actual` — true transaction class
- `Probability` — predicted fraud probability

The selected classification threshold converts probabilities into predicted classes:

```text
Probability ≥ Threshold → FRAUD
Probability < Threshold → NORMAL
```

The dashboard dynamically calculates:

| Metric | Description |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Percentage of predicted fraud transactions that are actually fraud |
| Recall | Percentage of actual fraud transactions correctly detected |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Model's ability to distinguish between fraud and normal transactions |
| PR-AUC | Precision–Recall performance, especially useful for imbalanced data |

Because fraud detection datasets are highly imbalanced, the project evaluates precision, recall, F1 Score, ROC-AUC, and PR-AUC in addition to accuracy.
---

## 🎯 Threshold Optimization

The dashboard allows the fraud classification threshold to be adjusted interactively.

It compares model performance across multiple threshold values and identifies the threshold with the highest F1 Score as the recommended threshold.

This demonstrates the trade-off between:

```text
Lower Threshold
    ↓
More fraud detected
Higher Recall
Possible increase in false positives

Higher Threshold
    ↓
Fewer false positives
Higher Precision
Possible missed fraud transactions
```
---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- SHAP
- Matplotlib
- Seaborn
- Jupyter Notebook
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
Interactive Streamlit Dashboard
```
---

## 📁 Project Structure

```text
fraud-detection-system/
│
├── app.py
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
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then be available locally in your browser.

---

## 🖥️ Dashboard Capabilities

### Single Transaction Prediction

Users can enter transaction feature values and receive:

- Predicted transaction class
- Fraud probability
- Classification threshold

### Model Explainability

The dashboard provides model explainability using SHAP.

For individual predictions, users can view:

- Prediction result
- Fraud probability
- Risk level
- Top 10 feature contributions
- SHAP values
- Direction of each feature's influence
  - Increases fraud risk
  - Reduces fraud risk
  - No significant impact

The dashboard also provides Random Forest feature importance visualization
showing the top 10 most important features used by the model.


### Batch Prediction

Users can upload a CSV file containing multiple transactions.

The dashboard provides:

- Predicted classes
- Fraud probabilities
- Total transaction count
- Normal transaction count
- Fraud transaction count
- Downloadable prediction results

### Real Model Evaluation

The dashboard evaluates the trained model using:

```text
real_model_evaluation.csv
```

It displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- ROC Curve
- Precision–Recall Curve
- Confusion Matrix

---

## 🧠 Model Explainability

The dashboard provides two complementary explainability methods.

### 📊 Feature Importance

The Random Forest model's built-in feature importance scores identify the features that contribute most strongly to the model's overall decision-making.

The dashboard displays:

- Feature importance visualization
- Top 10 most important features
- Feature importance scores

### 🔍 SHAP Explanations

For individual transaction predictions, SHAP (SHapley Additive exPlanations) is used to explain why the model produced a particular prediction.

For each prediction, the dashboard provides:

- Prediction result
- Fraud probability
- Risk level
- Top 10 feature contributions
- SHAP Value
- Direction of contribution:
  - Increases fraud risk
  - Reduces fraud risk
  - No significant impact

This provides transaction-level interpretability in addition to the global feature importance analysis.


## 📌 Future Improvements

- Real-time transaction prediction
- Database integration
- REST API development
- Docker containerization
- Cloud deployment
- Automated model monitoring

---

## 👨‍💻 Author

**Shubham Kumar**

GitHub: https://github.com/agarwalshubham-97
