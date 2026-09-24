import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# TASK 1: Load dataset ONCE and save offline fallback
print("--- TASK 1: Loading Dataset & Creating Offline Fallback ---")
df = sns.load_dataset('titanic')
fallback_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')
df.to_csv(fallback_path, index=False)
print(f"Saved fallback CSV to {fallback_path}")
print(f"Dataset Shape: {df.shape}")

# Missing value percentages
print("\n--- Missing Value Percentages ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_summary = missing_pct[missing_pct > 0]
print(missing_summary)

# TASK 2: Missing value handling decision logging
print("\n--- TASK 2: Missing Value Strategy Evaluation ---")
for col, pct in missing_summary.items():
    if pct < 5:
        strat = "Drop rows (<5% missing)"
    elif 5 <= pct <= 30:
        strat = "Impute with median/mode (5%-30% missing)"
    else:
        strat = "Drop column or encode 'Missing' category (>30% missing)"
    print(f"Column '{col}': {pct:.2f}% missing -> Strategy: {strat}")

# Clean copy for EDA
df_cleaned = df.copy()
df_cleaned = df_cleaned.dropna(subset=['embarked', 'embark_town'])
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].median())
df_cleaned['deck'] = df_cleaned['deck'].astype(str).fillna('Missing')

# TASK 3: Univariate Analysis & Outliers (Age & Fare)
print("\n--- TASK 3: Univariate Analysis ---")
for col in ['age', 'fare']:
    q1 = df_cleaned[col].quantile(0.25)
    q3 = df_cleaned[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df_cleaned[(df_cleaned[col] < (q1 - 1.5 * iqr)) | (df_cleaned[col] > (q3 + 1.5 * iqr))]
    print(f"Column '{col}': {len(outliers)} outliers based on IQR rule.")

fare_mean = df_cleaned['fare'].mean()
fare_median = df_cleaned['fare'].median()
fare_mode = df_cleaned['fare'].mode()[0]
print(f"Fare Stats - Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")

# TASK 4: Bivariate Survival Rates & Correlation Matrix
print("\n--- TASK 4: Bivariate Survival Rates ---")
print("Survival rate by Sex:\n", df_cleaned.groupby('sex')['survived'].mean())
print("\nSurvival rate by Pclass:\n", df_cleaned.groupby('pclass')['survived'].mean())
print("\nSurvival rate by Sex and Pclass:\n", df_cleaned.groupby(['sex', 'pclass'])['survived'].mean())

num_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_cleaned[num_cols].corr()
print("\n6x6 Correlation Matrix:\n", corr_matrix)

# TASK 6: Z-Score Standardization Check
print("\n--- TASK 6: Z-Score Standardization Check ---")
for col in ['age', 'fare']:
    z_col = (df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std()
    print(f"Standardized {col} -> Mean: {z_col.mean():.4f}, Std: {z_col.std():.4f}")

print("\n--- Part A Complete! ---")
