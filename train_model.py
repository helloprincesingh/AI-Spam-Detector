import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
data_path = "data/SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Clean dataset
df = df.dropna(subset=["label", "message"])
df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates(subset=["message"])

print("Dataset loaded successfully!")
print("\nDataset distribution:")
print(df["label"].value_counts())

# Convert labels into numeric values
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

X = df["message"]
y = df["label"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create NLP + Machine Learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save complete pipeline
joblib.dump(model, "models/spam_model.pkl")
# Save evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0))
}

joblib.dump(metrics, "models/metrics.pkl")
# Calculate and save confusion matrix
cm = confusion_matrix(y_test, predictions)

joblib.dump(cm, "models/confusion_matrix.pkl")

print("\nConfusion Matrix:")
print(cm)
print("\nEvaluation metrics saved successfully!")
print(metrics)
print("\nModel saved to models/spam_model.pkl")