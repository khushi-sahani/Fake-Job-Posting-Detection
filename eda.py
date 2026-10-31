
import pandas as pd
from pathlib import Path

# Find the dataset
file_path = Path("data/fake_job_postings.csv")

# Check whether the file exists
if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Check the data folder and filename."
    )

# Load the CSV
df = pd.read_csv(file_path)

# Basic dataset information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
df.info()

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sort_values(ascending=False))

print("\n--- FRAUD LABEL COUNTS ---")
print(df["fraudulent"].value_counts())

print("\n--- FRAUD LABEL PERCENTAGES ---")
print(
    (df["fraudulent"].value_counts(normalize=True) * 100).round(2)
)
