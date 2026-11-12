import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score

# Load the trained model
bundle = joblib.load("models/best_model.joblib")

model = bundle["pipeline"]
features = bundle["features"]

# Load the original dataset
df = pd.read_csv("data/student-por.csv", sep=";")

X = df.drop(columns=["G3"])
y = (df["G3"] >= 10).astype(int)

# Recreate the same training split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# We use training data to avoid further test-set tuning
fail_f1 = make_scorer(f1_score, pos_label=0, zero_division=0)

result = permutation_importance(
    model,
    X_train[features],
    y_train,
    scoring=fail_f1,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

importance = pd.DataFrame(
    {
        "Feature": features,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }
)

importance = importance.sort_values("Importance", ascending=False)

print("\nTOP 15 MOST IMPORTANT FEATURES")
print(importance.head(15).to_string(index=False))

importance.to_csv("outputs/feature_importance.csv", index=False)

top_features = importance.head(15).iloc[::-1]

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])

plt.xlabel("Decrease in Fail F1 After Shuffling")
plt.title("Random Forest Permutation Feature Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importance.png", dpi=150)

plt.close()

print("\nFeature importance analysis complete!")
