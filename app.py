
import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import re

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide"
)

# ---------------- PROJECT PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fake_job_detector.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.json"

# ---------------- LOAD MODEL AND METRICS ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔍 Fake Job Detector")

page = st.sidebar.radio(
    "Navigation",
    ["Job Detection", "Model Dashboard", "About Project"]
)

# ---------------- JOB DETECTION ----------------

if page == "Job Detection":

    st.title("🔍 Fake Job Posting Detection")

    st.write(
        "Analyze job postings using Machine Learning "
        "to identify potentially fraudulent opportunities."
    )

    st.divider()

    st.subheader("Enter Job Posting Details")

    title = st.text_input("Job Title")

    company_profile = st.text_area("Company Profile")

    description = st.text_area("Job Description")

    requirements = st.text_area("Job Requirements")

    benefits = st.text_area("Benefits")

    if st.button("Analyze Job Posting", type="primary"):

        if not title.strip() or not description.strip():

            st.warning(
                "Please enter at least the job title and description."
            )

        else:

            combined_text = " ".join([
                title,
                company_profile,
                description,
                requirements,
                benefits
            ])

            combined_text = re.sub(
                r"\s+",
                " ",
                combined_text.lower()
            ).strip()

            prediction = model.predict([combined_text])[0]

            fraud_probability = model.predict_proba(
                [combined_text]
            )[0][1]

            st.divider()

            st.subheader("Analysis Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:
                    st.error("⚠️ Potentially Fraudulent Job")

                else:
                    st.success("✅ Predicted as Legitimate")

            with col2:

                st.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )

            st.progress(float(fraud_probability))

            st.caption(
                "This is the model's estimated probability for the "
                "fraudulent class, not a guarantee of fraud."
            )

            if prediction == 1:

                st.warning(
                    "Verify the employer, contact details and job offer "
                    "independently. Never pay fees to secure a job."
                )

            else:

                st.info(
                    "A legitimate prediction does not guarantee that "
                    "the job or employer is genuine. Always verify "
                    "independently."
                )

# ---------------- MODEL DASHBOARD ----------------

elif page == "Model Dashboard":

    st.title("📊 Model Performance Dashboard")

    st.write(
        "Evaluation results from the held-out test dataset."
    )

    st.divider()

    # Performance metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    st.divider()

    # Model information

    st.subheader("Model Information")

    st.write("**Selected Algorithm:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")

    st.write(
        "**Training Samples:**",
        metrics["training_samples"]
    )

    st.write(
        "**Testing Samples:**",
        metrics["testing_samples"]
    )

    st.write(
        "**ROC-AUC:**",
        f"{metrics['roc_auc'] * 100:.2f}%"
    )

    st.divider()

    # Confusion matrix

    st.subheader("Confusion Matrix")

    confusion_data = pd.DataFrame(
        [[3369, 34], [16, 157]],
        index=["Actual Legitimate", "Actual Fraudulent"],
        columns=["Predicted Legitimate", "Predicted Fraudulent"]
    )

    st.dataframe(
        confusion_data,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_data.to_numpy(),
        display_labels=["Legitimate", "Fraudulent"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Rows represent actual classes and columns represent predictions."
    )

    st.divider()

    # Model comparison

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Naive Bayes"
        ],
        "Accuracy": [
            98.60,
            96.98
        ],
        "Precision": [
            82.20,
            81.55
        ],
        "Recall": [
            90.75,
            48.55
        ],
        "F1 Score": [
            86.26,
            60.87
        ],
        "ROC-AUC": [
            98.99,
            94.81
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Logistic Regression was selected because it achieved "
        "better fraud recall and F1-score in our model comparison."
    )

# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.write("""
    ### Fake Job Posting Detection System

    This project uses Natural Language Processing and
    Machine Learning to identify potentially fraudulent
    job advertisements.

    **Dataset:** Employment Scam Aegean Dataset (EMSCAD)

    **Total Job Postings:** 17,880

    **Machine Learning Algorithm:** Logistic Regression

    **Text Feature Extraction:** TF-IDF

    **Libraries Used:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Matplotlib
    - Joblib

    ### Limitations

    - Predictions depend on patterns learned from the training dataset.
    - New fraud techniques may not be recognized.
    - A prediction cannot establish whether an employer is genuine.
    - Human verification is always recommended.
    """)

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Fake Job Posting Detection System | "
    "Machine Learning & Natural Language Processing"
)
