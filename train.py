import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    ConfusionMatrixDisplay,
)

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("data/student-por.csv", sep=";")

# 0 = Fail, 1 = Pass
y = (df["G3"] >= 10).astype(int)

# Remove target grade from predictors
X = df.drop(columns=["G3"])

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 2. Create a single stratified train/test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training class distribution:")
print(y_train.value_counts())

# -------------------------------------------------
# 3. Define models
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# Two feature configurations
feature_sets = {
    "Early Prediction": X.drop(columns=["G1", "G2"]).columns.tolist(),
    "With Previous Grades": X.columns.tolist(),
}

# Fail is label 0, so evaluate it explicitly
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "fail_precision": make_scorer(precision_score, pos_label=0, zero_division=0),
    "fail_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
    "fail_f1": make_scorer(f1_score, pos_label=0, zero_division=0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_pipeline(model, features):
    categorical = (
        X[features]
        .select_dtypes(include=["object", "str", "category"])
        .columns.tolist()
    )

    numerical = [col for col in features if col not in categorical]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical),
        ]
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


# -------------------------------------------------
# 4. Cross-validation on training data only
# -------------------------------------------------
cv_results = []

for feature_name, features in feature_sets.items():
    print(f"\n=== {feature_name} ===")

    for model_name, model in models.items():
        pipeline = make_pipeline(model, features)

        scores = cross_validate(
            pipeline, X_train[features], y_train, cv=cv, scoring=scoring, n_jobs=-1
        )

        result = {"Feature Set": feature_name, "Model": model_name}

        for metric in scoring:
            result[f"CV {metric}"] = np.mean(scores[f"test_{metric}"])

        cv_results.append(result)

        print(
            f"{model_name}: "
            f"Fail F1 = {result['CV fail_f1']:.3f}, "
            f"Balanced Accuracy = "
            f"{result['CV balanced_accuracy']:.3f}"
        )

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

# -------------------------------------------------
# 5. Select the best model using CV Fail F1
# -------------------------------------------------
best_row = cv_df.sort_values(
    by=["CV fail_f1", "CV balanced_accuracy"], ascending=False
).iloc[0]

best_name = best_row["Model"]
best_feature_set = best_row["Feature Set"]
best_features = feature_sets[best_feature_set]

print("\nBEST MODEL FROM CROSS-VALIDATION")
print("Algorithm:", best_name)
print("Feature set:", best_feature_set)
print("CV Fail F1:", round(best_row["CV fail_f1"], 3))

# -------------------------------------------------
# 6. Fit each model and evaluate on test data
# -------------------------------------------------
test_results = []

for feature_name, features in feature_sets.items():
    for model_name, model in models.items():
        pipeline = make_pipeline(model, features)

        pipeline.fit(X_train[features], y_train)
        predictions = pipeline.predict(X_test[features])

        result = {
            "Feature Set": feature_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
            "Fail Precision": precision_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "Fail Recall": recall_score(
                y_test, predictions, pos_label=0, zero_division=0
            ),
            "Fail F1": f1_score(y_test, predictions, pos_label=0, zero_division=0),
        }

        test_results.append(result)

        # Save the preselected best pipeline
        if model_name == best_name and feature_name == best_feature_set:
            joblib.dump(
                {
                    "pipeline": pipeline,
                    "features": features,
                    "feature_set": feature_name,
                    "model_name": model_name,
                },
                MODEL_DIR / "best_model.joblib",
            )

            print("\nBEST MODEL TEST REPORT")
            print(
                classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1],
                    target_names=["Fail", "Pass"],
                    zero_division=0,
                )
            )

            cm = confusion_matrix(y_test, predictions, labels=[0, 1])

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Fail", "Pass"]
            )
            disp.plot(cmap="Blues")
            plt.title("Best Model Confusion Matrix")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
            plt.close()

# -------------------------------------------------
# 7. Majority-class baseline
# -------------------------------------------------
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

test_results.append(
    {
        "Feature Set": "Baseline",
        "Model": "Always Predict Pass",
        "Accuracy": accuracy_score(y_test, baseline_predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, baseline_predictions),
        "Fail Precision": precision_score(
            y_test, baseline_predictions, pos_label=0, zero_division=0
        ),
        "Fail Recall": recall_score(
            y_test, baseline_predictions, pos_label=0, zero_division=0
        ),
        "Fail F1": f1_score(y_test, baseline_predictions, pos_label=0, zero_division=0),
    }
)

# -------------------------------------------------
# 8. Save comparison table and chart
# -------------------------------------------------
results_df = pd.DataFrame(test_results)

print("\nFINAL TEST RESULTS")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

plt.figure(figsize=(11, 6))

chart_df = results_df[results_df["Feature Set"] != "Baseline"]

sns.barplot(data=chart_df, x="Model", y="Fail F1", hue="Feature Set")

plt.title("Model Comparison: Fail F1-Score")
plt.ylabel("Fail F1-Score")
plt.ylim(0, 1)
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
plt.close()

print("\nTRAINING COMPLETE!")
print("Results saved in:", OUTPUT_DIR.resolve())
print("Best model saved in:", MODEL_DIR.resolve())
