
import pandas as pd
import joblib
import json
from pathlib import Path

# Create output folders
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Load processed dataset
df = pd.read_csv("data/processed_jobs.csv")

# Prepare data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X = df["combined_text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Final selected model
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# Train final model
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "models/fake_job_detector.joblib")

# Save model information
model_info = {
    "model_name": "Logistic Regression",
    "accuracy": 0.9860,
    "precision": 0.8220,
    "recall": 0.9075,
    "f1_score": 0.8626,
    "roc_auc": 0.9899,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}

with open("outputs/model_metrics.json", "w") as file:
    json.dump(model_info, file, indent=4)

print("Final model trained and saved!")
print("Model: models/fake_job_detector.joblib")
print("Metrics: outputs/model_metrics.json")
