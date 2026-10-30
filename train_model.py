
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# FRAUDSHIELD AI - MODEL TRAINING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Place creditcard.csv inside the data folder."
    )

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "Class" not in df.columns:
    raise ValueError("Dataset must contain a 'Class' column.")

if df["Class"].isna().any():
    raise ValueError("The target column contains missing values.")

df = df.dropna().copy()

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Class must contain 0 for legitimate and 1 for fraud.")

print(f"Dataset shape: {df.shape}")
print(f"Legitimate transactions: {(y == 0).sum():,}")
print(f"Fraudulent transactions: {(y == 1).sum():,}")

# Keep the same feature columns for future predictions.
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)
    trained_models[name] = model

    print(f"\n{name} results:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))

    # Save a confusion matrix for each model.
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{filename}_confusion_matrix.png")
    plt.close()

# Compare models using F1-score.
results_df = pd.DataFrame(results).sort_values(
    "F1-Score", ascending=False
)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

# Save the fitted pipeline and the exact feature order.
joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_columns,
        "model_name": best_name,
    },
    BASE_DIR / "fraudshield_model.joblib",
)

print("\nModel comparison:")
print(results_df.to_string(index=False))
print(f"\nSelected model by test F1-score: {best_name}")
print(f"Saved model: {BASE_DIR / 'fraudshield_model.joblib'}")
print(f"Saved reports and charts in: {OUTPUT_DIR}")
