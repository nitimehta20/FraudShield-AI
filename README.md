# FraudShield AI
### Intelligent Financial Fraud Detection System

FraudShield AI is a machine learning-based application designed to identify potentially fraudulent financial transactions. It provides an interactive Streamlit dashboard where users can explore transaction data, compare machine learning models, and assess transaction risk.

## Features

- **Fraud Detection:** Uses machine learning models to classify potentially fraudulent transactions.
- **Model Comparison:** Compares Logistic Regression and Random Forest.
- **Performance Evaluation:** Displays Precision, Recall, F1-score and ROC-AUC where available.
- **Confusion Matrix:** Helps evaluate model classification performance.
- **Transaction Risk Scanner:** Provides risk assessment for individual transactions.
- **Interactive Dashboard:** Presents model results and visual insights using Streamlit and Plotly.
- **Class Imbalance Handling:** Addresses the imbalance between fraudulent and legitimate transactions during model development.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Project Structure

```text
FraudShield-AI/
│
├── app.py
├── fraudshield_model.joblib
├── requirements.txt
├── README.md
│
├── data/
│   └── creditcard.csv
│
└── outputs/
```

*Note: The structure above is illustrative. Files and folders may vary depending on your local project setup. The dataset and generated model files may be excluded from the public repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nitimehta20/FraudShield-AI.git
```

### 2. Navigate to the project folder

```bash
cd FraudShield-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the dataset

Place the required `creditcard.csv` dataset in the `data/` folder or in the project root, according to the application's configured data paths.

### 7. Run the application

```bash
streamlit run app.py
```

## Machine Learning Approach

The project explores machine learning techniques for identifying fraudulent transactions.

The models include:

1. Logistic Regression
2. Random Forest Classifier

The models are evaluated using classification metrics such as precision, recall, F1-score and ROC-AUC. These metrics help assess how effectively the models identify fraudulent transactions, particularly when the dataset contains imbalanced classes.

## Project Objective

The objective of FraudShield AI is to demonstrate how machine learning can support financial fraud detection through data processing, model evaluation, risk assessment and interactive visualization.

## Limitations

- This project is an educational prototype.
- It uses a public anonymized dataset.
- It is not connected to live banking or payment systems.
- Predictions should not be used for real financial decisions without extensive testing, validation and security review.

## Author

**Niti Mehta**

B.Sc. Data Science and Business Analytics

GitHub: [nitimehta20](https://github.com/nitimehta20)

---

*FraudShield AI – Exploring smarter financial fraud detection through machine learning.*