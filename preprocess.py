
import pandas as pd
from pathlib import Path

# Load dataset
file_path = Path("data/fake_job_postings.csv")
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Columns containing text
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Fill missing text with empty strings
for column in text_columns:
    df[column] = df[column].fillna("")

# Categorical columns
categorical_columns = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "salary_range"
]

# Fill missing categorical values
for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
output_path = Path("data/cleaned_jobs.csv")
df.to_csv(output_path, index=False)

print("\nCleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("\nSaved to:", output_path)
