import pandas as pd

# Load dataset
df = pd.read_csv("D:\Sparsh\AI_Projects\AI_Fake_Review_And_Content_Detector\Datasets\fake reviews dataset.csv")

print("Original Columns:", df.columns)

# Keep only required columns
df = df[["text_", "label"]]  # change if column name different

# Rename text column
df.rename(columns={"text_": "text"}, inplace=True)

# Convert labels
# If labels are OR / CG
df["label"] = df["label"].map({"OR": 0, "CG": 1})

# Drop missing
df.dropna(inplace=True)

# Save clean dataset
df.to_csv("Dataset/reviews_clean.csv", index=False)

print("Clean dataset saved.")
print(df["label"].value_counts())
