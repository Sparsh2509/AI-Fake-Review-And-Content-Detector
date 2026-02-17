import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"D:\Sparsh\AI_Projects\AI_Fake_Review_And_Content_Detector\Datasets\reviews_ai_human.csv")

# Drop nulls if any
df.dropna(inplace=True)

X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Print results
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "models/ai_model.pkl")
joblib.dump(vectorizer, "models/ai_vectorizer.pkl")

print("✅ AI Detector Saved.")
