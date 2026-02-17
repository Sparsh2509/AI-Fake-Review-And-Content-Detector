import pandas as pd

df = pd.read_csv(r"D:\Sparsh\AI_Projects\AI_Fake_Review_And_Content_Detector\Datasets\reviews_ai_human.csv")
print(df["label"].value_counts())
print(df.isnull().sum())
