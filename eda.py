import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/student-por.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# UCI student CSV files use semicolons, not commas
df = pd.read_csv(DATA_PATH, sep=";")

print("DATASET SHAPE")
print(df.shape)

print("\nFIRST FIVE RECORDS")
print(df.head())

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY")
print(df.describe())


# Define student performance
# 1 = Pass, 0 = Fail
df["pass"] = (df["G3"] >= 10).astype(int)

print("\nPASS / FAIL DISTRIBUTION")
print(df["pass"].value_counts())

print("\nPASS / FAIL PERCENTAGES")
print(df["pass"].value_counts(normalize=True) * 100)

print("\nGRADE STATISTICS")
print(df[["G1", "G2", "G3"]].describe())


sns.set_theme(style="whitegrid")

# 1. Final grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=21, discrete=True)
plt.axvline(10, color="red", linestyle="--", label="Pass threshold = 10")
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "grade_distribution.png")
plt.close()

# 2. Pass vs Fail distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="pass", order=[0, 1])
plt.xticks([0, 1], ["Fail", "Pass"])
plt.title("Pass vs Fail Distribution")
plt.xlabel("Final Performance")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pass_fail.png")
plt.close()

# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time Category")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "studytime_grades.png")
plt.close()

# 4. Previous failures vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="failures", y="G3")
plt.title("Previous Failures vs Final Grade")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "failures_grades.png")
plt.close()

# 5. Correlation heatmap for numeric variables
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(13, 10))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("\nCORRELATION WITH FINAL GRADE")
print(numeric_df.corr()["G3"].sort_values(ascending=False))

print("\nEDA COMPLETE!")
print("Graphs saved in:", OUTPUT_DIR.resolve())
