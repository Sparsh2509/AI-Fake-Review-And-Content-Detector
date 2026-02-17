import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load clean dataset
df = pd.read_csv(r"D:\Sparsh\AI_Projects\AI_Fake_Review_And_Content_Detector\Datasets\reviews_clean.csv")

X = df["text"]
y = df["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/fake_model.joblib")
joblib.dump(vectorizer, "models/fake_vectorizer.joblib")

print("Fake Review Model Saved.")
