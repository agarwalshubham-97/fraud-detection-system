# 💳 Credit Card Fraud Detection System

A machine learning based credit card fraud detection system built using Python, Scikit-learn, and Streamlit.

The project uses a Random Forest classifier to identify potentially fraudulent credit card transactions and provides both single-transaction and batch CSV prediction through an interactive Streamlit dashboard.

## 🚀 Features

- Credit card transaction fraud detection
- Random Forest classification model
- Highly imbalanced dataset handling
- Fraud probability prediction
- Configurable fraud detection threshold
- Single transaction prediction
- Batch CSV transaction prediction
- Transaction summary
- Fraud probability visualization
- Downloadable prediction results
- Interactive Streamlit dashboard

## 📊 Model Performance

The final Random Forest model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 94.12% |
| Recall | 81.63% |
| F1 Score | 87.43% |
| ROC-AUC | 97.98% |
| PR-AUC | 86.80% |

Because credit card fraud datasets are highly imbalanced, precision, recall, F1-score, ROC-AUC, and PR-AUC are considered along with accuracy.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## 📁 Project Structure

```text
fraud-detection-system/
│
├── app.py
├── README.md
├── requirements.txt
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
├── test_mixed_transactions.csv
└── test_transactions.csv