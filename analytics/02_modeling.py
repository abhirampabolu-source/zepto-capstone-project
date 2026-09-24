import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load offline fallback dataset
csv_path = os.path.join(os.path.dirname(__file__), 'titanic.csv')
df = pd.read_csv(csv_path)

# Drop redundant or high-cardinality/leaky columns
df_model = df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'adult_male'], errors='ignore')

# Separate Features & Target
X = df_model.drop(columns=['survived'])
y = df_model['survived']

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Define Preprocessing Pipelines
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# TASK 7: Stratified Train/Test Split
print("--- TASK 7: Stratified Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
print("Train Class Dist:\n", y_train.value_counts(normalize=True))

# TASK 8 & 9: Model Training & Evaluation
print("\n--- TASK 8 & 9: Classification Models Evaluation ---")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

for name, clf in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

# TASK 10: Handling Class Imbalance (Baseline vs Weighted vs SMOTE)
print("\n--- TASK 10: Class Imbalance Techniques (Random Forest) ---")
rf_weighted = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])
rf_weighted.fit(X_train, y_train)
y_pred_w = rf_weighted.predict(X_test)
print(f"Weighted RF - F1: {f1_score(y_test, y_pred_w):.4f}, Recall: {recall_score(y_test, y_pred_w):.4f}")

rf_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
rf_smote.fit(X_train, y_train)
y_pred_s = rf_smote.predict(X_test)
print(f"SMOTE RF    - F1: {f1_score(y_test, y_pred_s):.4f}, Recall: {recall_score(y_test, y_pred_s):.4f}")

# TASK 11: Hyperparameter Tuning
print("\n--- TASK 11: GridSearchCV Hyperparameter Tuning ---")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, oob_score=True))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
oob = best_model.named_steps['classifier'].oob_score_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
print(f"OOB Score: {oob:.4f}")

# TASK 12: Regression Task (Predicting Fare)
print("\n--- TASK 12: Regression Task (Predicting Fare) ---")
X_reg = df_model.drop(columns=['fare'])
y_reg = df_model['fare']

num_cols_reg = X_reg.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_reg = X_reg.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

reg_preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols_reg),
    ('cat', cat_transformer, cat_cols_reg)
])

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.20, random_state=42)

reg_pipe = Pipeline([
    ('preprocessor', reg_preprocessor),
    ('regressor', LinearRegression())
])
reg_pipe.fit(X_tr_r, y_tr_r)
y_pred_r = reg_pipe.predict(X_te_r)

mae = mean_absolute_error(y_te_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
r2 = r2_score(y_te_r, y_pred_r)
n, p = X_te_r.shape[0], X_te_r.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Adj R2: {adj_r2:.4f}")

# TASK 13: Pipeline Export
print("\n--- TASK 13: Exporting Fitted Pipeline ---")
model_path = os.path.join(os.path.dirname(__file__), 'pipeline.joblib')
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Verification reload
loaded_model = joblib.load(model_path)
print("Pipeline reload test passed successfully!")

print("\n--- Part B Complete! ---")
