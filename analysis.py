import pandas as pd
import matplotlib.pyplot as plt

# خواندن دیتا
df = pd.read_csv("data/aug_train.csv")

# دیدن 5 سطر اول
print(df.head())

# شکل دیتا (چند سطر و ستون)
print(df.shape)

# اسم ستون‌ها
print(df.columns)

# اطلاعات کلی دیتا
print(df.info())

# بررسی هدف (خیلی مهم)
print(df["target"].value_counts())

print(df.isnull().sum())
print(df.describe())
df["target_meaning"] = df["target"].map({0: "Stay", 1: "Change Job"})
print(df["target_meaning"].value_counts())
print(df.groupby("education_level")["target"].mean())
df.groupby("gender")["target"].mean()
df.groupby("experience")["target"].mean().sort_values(ascending=False)
df.groupby("training_hours")["target"].mean()

df["target"].value_counts().plot(kind="bar")
plt.title("Job Change Distribution")
plt.show()

print("Percentage of each class:")
print(df["target"].value_counts(normalize=True) * 100)

print(df.groupby("experience")["target"].mean().sort_values(ascending=False))

