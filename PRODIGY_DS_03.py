import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import math

# Step 1: Data Collection
df = pd.read_csv("social_media_vs_productivity.csv")  # Use your file path here
print(df)
# Step 2: Data Cleaning & Preprocessing
df_clean = df.dropna(subset=['actual_productivity_score']).copy()
print(df_clean)
num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
print(num_cols)
imputer = SimpleImputer(strategy='median')
print(imputer)
df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])

cat_cols = df_clean.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# Step 3: Exploratory Data Analysis (More detailed)

# 3.1 Correlation Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3.2 Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df_clean['actual_productivity_score'], kde=True, bins=30, color='purple')
plt.title("Distribution of Actual Productivity Score")
plt.xlabel("Actual Productivity Score")
plt.ylabel("Frequency")
plt.show()

# 3.3 Boxplots for numeric features to check outliers - DYNAMIC GRID
num_features = len(num_cols.drop('actual_productivity_score'))
cols = 3
rows = math.ceil(num_features / cols)

plt.figure(figsize=(cols*5, rows*4))
for i, col in enumerate(num_cols.drop('actual_productivity_score')):
    plt.subplot(rows, cols, i+1)
    sns.boxplot(y=df_clean[col], color='skyblue')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 3.4 Pairplot for a subset of features + target (use smaller subset for performance)
sample_features = num_cols.drop('actual_productivity_score').tolist()[:4] + ['actual_productivity_score']
sns.pairplot(df_clean[sample_features])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Step 4: Feature Engineering & Selection
X = df_clean.drop('actual_productivity_score', axis=1)
y = df_clean['actual_productivity_score']

# Step 5: Model Selection
model = RandomForestRegressor(random_state=42)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Evaluation (Baseline)
model.fit(X_train, y_train)
baseline_pred = model.predict(X_test)
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
baseline_r2 = r2_score(y_test, baseline_pred)

print(f"🔹 Baseline RMSE: {baseline_rmse:.2f}")
print(f"🔹 Baseline R² Score: {baseline_r2:.2f}")

# Plot Feature Importances from baseline model
importances = model.feature_importances_
features = X.columns
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title("Feature Importance (Baseline Model)")
plt.tight_layout()
plt.show()

# Step 8: Performance Tuning with Pipeline and GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Final Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_rmse = mean_squared_error(y_test, y_pred, squared=False)
final_r2 = r2_score(y_test, y_pred)

print("\n✅ Final Model Evaluation:")
print("Best Parameters:", grid_search.best_params_)
print(f"Final RMSE: {final_rmse:.2f}")
print(f"Final R² Score: {final_r2:.2f}")

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Step 9: Monitoring & Maintenance - Save the model
joblib.dump(best_model, "final_productivity_model.pkl")
print("\nModel saved as 'final_productivity_model.pkl'")
