import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Load cleaned fake-review dataset
df = pd.read_csv(r"D:\Sparsh\AI_Projects\AI_Fake_Review_And_Content_Detector\Datasets\reviews_clean.csv")

# Take only REAL reviews (label = 0)
df_real = df[df["label"] == 0].sample(500, random_state=42)

ai_reviews = []

for idx, row in df_real.iterrows():
    prompt = f"""
Rewrite this product review in a highly polished, balanced,
slightly formal AI-style tone. Keep meaning same.

Review:
{row['text']}
"""

    try:
        response = model.generate_content(prompt)
        ai_reviews.append(response.text.strip())
        print(f"Generated: {len(ai_reviews)}")
        sleep(1)  # Avoid rate limit

    except Exception as e:
        print("Error:", e)
        ai_reviews.append(row["text"])

# Create Human dataframe
human_df = df_real[["text"]].copy()
human_df["label"] = 0

# Create AI dataframe
ai_df = pd.DataFrame({
    "text": ai_reviews,
    "label": 1
})

# Combine both
final_df = pd.concat([human_df, ai_df])

# Save dataset
final_df.to_csv("Datasets/reviews_ai_human.csv", index=False)

print("✅ AI Dataset Created Successfully.")
