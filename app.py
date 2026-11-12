import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

st.set_page_config(
    page_title="Student Performance Prediction", page_icon="🎓", layout="wide"
)

MODEL_PATH = Path("models/best_model.joblib")
DATA_PATH = Path("data/student-por.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


bundle = load_model()
df = load_data()

model = bundle["pipeline"]
features = bundle["features"]

st.title("🎓 Student Performance Prediction")
st.write("Machine Learning Classification Project | UCI Student Performance Dataset")

st.info(
    f"Selected model: {bundle['model_name']} | "
    f"Feature configuration: {bundle['feature_set']}"
)

st.markdown("---")

st.subheader("Enter Student Information")
st.caption(
    "Default values are based on the dataset. Change the fields to simulate a student."
)

# Important academic inputs shown first
priority_features = ["G1", "G2", "studytime", "failures", "absences", "age"]

priority_features = [f for f in priority_features if f in features]

other_features = [f for f in features if f not in priority_features]

labels = {
    "G1": "First Period Grade (G1)",
    "G2": "Second Period Grade (G2)",
    "studytime": "Weekly Study Time",
    "failures": "Previous Class Failures",
    "absences": "Number of Absences",
    "age": "Student Age",
}


def make_input(feature):
    series = df[feature]
    label = labels.get(feature, feature.replace("_", " ").title())

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        options = sorted(series.dropna().unique().tolist())
        default = series.mode().iloc[0]

        return st.selectbox(label, options, index=options.index(default), key=feature)

    # All remaining features in this dataset are integers
    minimum = int(series.min())
    maximum = int(series.max())
    default = int(series.median())

    return st.number_input(
        label, min_value=minimum, max_value=maximum, value=default, step=1, key=feature
    )


student = {}

with st.form("student_form"):
    st.markdown("#### Academic Information")

    col1, col2 = st.columns(2)

    for i, feature in enumerate(priority_features):
        with col1 if i % 2 == 0 else col2:
            student[feature] = make_input(feature)

    with st.expander("Additional Student Information"):
        col3, col4 = st.columns(2)

        for i, feature in enumerate(other_features):
            with col3 if i % 2 == 0 else col4:
                student[feature] = make_input(feature)

    submitted = st.form_submit_button(
        "Predict Student Performance", type="primary", use_container_width=True
    )

if submitted:
    # Ensure columns match training order
    input_df = pd.DataFrame([student])[features]

    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]
    classes = model.named_steps["classifier"].classes_

    fail_index = list(classes).index(0)
    pass_index = list(classes).index(1)

    fail_probability = float(probabilities[fail_index])
    pass_probability = float(probabilities[pass_index])

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Outcome: PASS")
    else:
        st.error("Predicted Outcome: FAIL")

    left, right = st.columns(2)

    with left:
        st.metric("Estimated Pass Probability", f"{pass_probability:.1%}")

    with right:
        st.metric("Estimated Fail Probability", f"{fail_probability:.1%}")

    st.progress(pass_probability)

    st.caption(
        "Probabilities are model estimates, not guarantees. "
        "This is an educational prototype and should not "
        "be used for actual academic decisions."
    )

    with st.expander("View Model Input"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "Algorithms compared: Logistic Regression, "
    "Decision Tree, and Random Forest. "
    "Dataset: UCI Student Performance (Portuguese)."
)
