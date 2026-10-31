# Fake Job Posting Detection System

A Machine Learning and Natural Language Processing project that identifies potentially fraudulent job postings using text classification.

## Project Overview

Online job platforms can contain misleading or fraudulent job advertisements. This project uses machine learning to analyze job posting information and predict whether a posting is potentially fraudulent or legitimate.

## Dataset

- **Dataset:** Employment Scam Aegean Dataset (EMSCAD)
- **Total Job Postings:** 17,880
- **Legitimate Postings:** 17,014
- **Fraudulent Postings:** 866

Dataset source: [Kaggle – Real or Fake Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-job-posting-prediction)

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Naive Bayes
- Streamlit
- Matplotlib
- Joblib

## Model Performance

The selected Logistic Regression model was evaluated on a held-out test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 98.60% |
| Precision | 82.20% |
| Recall | 90.75% |
| F1 Score | 86.26% |
| ROC-AUC | 98.99% |

Logistic Regression was selected after comparison with Multinomial Naive Bayes.

## Project Structure

```text
Fake-Job-Posting-Detection/
│
├── data/
│   ├── fake_job_postings.csv
│   ├── cleaned_jobs.csv
│   └── processed_jobs.csv
│
├── models/
│   └── fake_job_detector.joblib
│
├── outputs/
│   └── model_metrics.json
│
├── app.py
├── eda.py
├── preprocess.py
├── train_model.py
├── compare_models.py
├── finalize_model.py
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Fake-Job-Posting-Detection.git
```

2. Open the project folder:

```bash
cd Fake-Job-Posting-Detection
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
streamlit run app.py
```

## Application Features

- Analyze job postings through a simple web interface.
- Predict potentially fraudulent or legitimate postings.
- Display estimated fraud probability.
- View model performance metrics.
- Visualize the confusion matrix.
- Compare Logistic Regression and Naive Bayes.

## Limitations

- Predictions depend on patterns learned from the training dataset.
- New fraud techniques may not be recognized.
- The dataset has an imbalanced class distribution.
- A model prediction cannot guarantee whether an employer is genuine.

Users should independently verify employers and job offers.

## Disclaimer

This project is developed for educational and demonstration purposes. It should not be used as the sole basis for employment or financial decisions.