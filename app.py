import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_fail_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "student_exam_performance.csv")

# These are the same input features used in the supplied IPython notebook.
FEATURE_COLS = [
    "age", "gender", "education_level", "school_type", "family_income",
    "parent_education", "urban_rural", "previous_exam_score", "previous_gpa",
    "attendance_percentage", "assignment_completion_rate",
    "class_participation", "study_hours_per_day", "self_study_hours",
    "private_tuition", "online_learning_hours", "study_consistency",
    "study_environment", "study_method", "revision_frequency",
    "practice_tests_completed", "notes_quality", "sleep_hours", "sleep_quality",
    "daily_screen_time", "physical_activity_hours", "break_frequency",
    "stress_level", "motivation_level", "internet_access", "device_availability",
    "educational_app_usage", "online_course_hours", "exam_difficulty",
    "exam_preparation_days", "time_management_score", "exam_anxiety_level"
]

@st.cache_resource
def load_model():
    # The pickle contains the Pipeline saved by the original notebook.
    # Only load pickle files you trust.
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

st.title("🎓 Student Pass/Fail Prediction")
st.write(
    "Adjust any feature below. The prediction and probability update automatically "
    "when a value changes."
)

try:
    model = load_model()
except Exception as exc:
    st.error("The pickle model could not be loaded.")
    st.markdown(
        "This app expects the model saved from your original notebook. "
        "The supplied pickle appears to have been created with scikit-learn 1.6.1. "
        "Install the versions listed in `requirements.txt`, then run the app again."
    )
    st.code("python -m pip install -r requirements.txt")
    st.exception(exc)
    st.stop()

try:
    df = load_dataset()
except Exception as exc:
    st.error("Could not read student_exam_performance.csv. Keep it beside app.py.")
    st.exception(exc)
    st.stop()

missing = [col for col in FEATURE_COLS if col not in df.columns]
if missing:
    st.error(f"Dataset is missing required feature columns: {missing}")
    st.stop()

available_features = df[FEATURE_COLS].copy()
numeric_features = available_features.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in FEATURE_COLS if c not in numeric_features]

# Use a representative row from the supplied dataset as the initial student.
default_row = available_features.iloc[0].copy()

with st.sidebar:
    st.header("About")
    st.write("Model: selected model from your original notebook")
    st.write(f"Input features: {len(FEATURE_COLS)}")
    st.caption(
        "The app uses the exact feature list from your notebook and does not "
        "retrain or replace your uploaded model."
    )

st.subheader("Student features")
st.caption("Change a value to recalculate the prediction. No submit button is required.")

values = {}
sections = [
    ("Background", [
        "age", "gender", "education_level", "school_type", "family_income",
        "parent_education", "urban_rural"
    ]),
    ("Academic history and attendance", [
        "previous_exam_score", "previous_gpa", "attendance_percentage",
        "assignment_completion_rate", "class_participation"
    ]),
    ("Study habits", [
        "study_hours_per_day", "self_study_hours", "private_tuition",
        "online_learning_hours", "study_consistency", "study_environment",
        "study_method", "revision_frequency", "practice_tests_completed",
        "notes_quality", "exam_preparation_days", "time_management_score"
    ]),
    ("Wellbeing and access", [
        "sleep_hours", "sleep_quality", "daily_screen_time",
        "physical_activity_hours", "break_frequency", "stress_level",
        "motivation_level", "internet_access", "device_availability",
        "educational_app_usage", "online_course_hours", "exam_difficulty",
        "exam_anxiety_level", "physical_activity_hours"
    ])
]

for section_name, section_cols in sections:
    section_cols = [c for c in section_cols if c in FEATURE_COLS]
    # Avoid duplicate widgets if a feature appears in multiple sections.
    section_cols = list(dict.fromkeys(section_cols))
    if not section_cols:
        continue
    st.markdown(f"### {section_name}")
    cols = st.columns(3)
    for i, col in enumerate(section_cols):
        with cols[i % 3]:
            label = col.replace("_", " ").title()
            if col in numeric_features:
                series = pd.to_numeric(available_features[col], errors="coerce").dropna()
                low = float(series.min()) if len(series) else 0.0
                high = float(series.max()) if len(series) else 100.0
                default = pd.to_numeric(pd.Series([default_row[col]]), errors="coerce").iloc[0]
                default = float(default) if pd.notna(default) else (low + high) / 2
                if low == high:
                    high = low + 1.0
                default = min(max(default, low), high)
                step = 1.0 if col in [
                    "age", "practice_tests_completed", "exam_preparation_days",
                    "stress_level", "private_tuition", "internet_access"
                ] else 0.1
                values[col] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    step=float(step),
                    key=f"feature_{col}"
                )
            else:
                options = available_features[col].dropna().unique().tolist()
                options = sorted(options, key=lambda x: str(x))
                if not options:
                    options = ["Unknown"]
                default = default_row[col]
                try:
                    index = options.index(default)
                except ValueError:
                    index = 0
                values[col] = st.selectbox(
                    label, options, index=index, key=f"feature_{col}"
                )

input_df = pd.DataFrame(
    [{col: values.get(col, default_row[col]) for col in FEATURE_COLS}],
    columns=FEATURE_COLS
)

try:
    predicted_status = str(model.predict(input_df)[0])
    pass_probability = None
    fail_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = [str(c) for c in model.classes_]
        probability_map = dict(zip(classes, probabilities))
        if "Pass" in probability_map:
            pass_probability = float(probability_map["Pass"])
        if "Fail" in probability_map:
            fail_probability = float(probability_map["Fail"])

    # Baseline is the first row from the dataset, using the same trained model.
    baseline_df = pd.DataFrame([default_row.to_dict()], columns=FEATURE_COLS)
    baseline_pass_probability = None
    if hasattr(model, "predict_proba"):
        base_probabilities = model.predict_proba(baseline_df)[0]
        base_classes = [str(c) for c in model.classes_]
        base_map = dict(zip(base_classes, base_probabilities))
        baseline_pass_probability = base_map.get("Pass")

    st.divider()
    st.subheader("Prediction result")
    first, second, third = st.columns(3)
    first.metric("Predicted status", predicted_status.upper())

    if pass_probability is not None:
        delta = None
        if baseline_pass_probability is not None:
            delta = f"{(pass_probability - baseline_pass_probability) * 100:+.3f} percentage points vs baseline"
        second.metric(
            "Pass probability",
            f"{pass_probability * 100:.3f}%",
            delta=delta
        )
        if fail_probability is not None:
            third.metric("Fail probability", f"{fail_probability * 100:.3f}%")
        st.progress(
            min(max(pass_probability, 0.0), 1.0),
            text=f"Estimated Pass probability: {pass_probability * 100:.3f}%"
        )
    else:
        second.info("This model does not provide probability estimates.")
        third.info("Predicted class only")

    if predicted_status.strip().lower() == "pass":
        st.success("The model predicts PASS.")
    else:
        st.error("The model predicts FAIL.")

    st.caption(
        "Small feature changes may cause small probability changes, but the probability "
        "may also remain the same due to the trained model's behavior and rounding. "
        "A 1-point attendance change will not always change the Pass/Fail label."
    )
    with st.expander("View input values sent to the model"):
        st.dataframe(input_df, use_container_width=True)

except Exception as exc:
    st.error(
        "Prediction failed. The pickle model and this app must use the same feature "
        "columns and compatible scikit-learn versions."
    )
    st.exception(exc)
