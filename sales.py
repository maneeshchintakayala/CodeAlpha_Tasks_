# ==========================================================
# SALES PREDICTION USING PYTHON
# Dataset: Advertising.csv
# ==========================================================

# ===========================
# 1. Import Required Libraries
# ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

# ===========================
# 2. Load Dataset
# ===========================

df = pd.read_csv("Advertising.csv")

print("="*60)
print("SALES PREDICTION USING MACHINE LEARNING")
print("="*60)

print("\nFirst Five Rows")
print(df.head())

# ===========================
# 3. Remove Unnecessary Column
# ===========================

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ===========================
# 4. Dataset Information
# ===========================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ===========================
# 5. Correlation Matrix
# ===========================

print("\nCorrelation Matrix")
print(df.corr())

# ===========================
# 6. Feature Selection
# ===========================

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

print("\nInput Features")
print(X.head())

print("\nTarget Variable")
print(y.head())

# ===========================
# 7. Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

# ===========================
# 8. Train Linear Regression
# ===========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ===========================
# 9. Model Coefficients
# ===========================

print("\nModel Coefficients")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

print("\nIntercept")
print(model.intercept_)

# ===========================
# 10. Prediction
# ===========================

y_pred = model.predict(X_test)

print("\nPredicted Sales")
print(y_pred[:10])

# ===========================
# 11. Evaluation Metrics
# ===========================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("---------------------------")
print("Mean Absolute Error :", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# ===========================
# 12. Actual vs Predicted Plot
# ===========================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.grid(True)

plt.show()

# ===========================
# 13. Feature Importance
# ===========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.abs(model.coef_)
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ===========================
# 14. Feature Importance Plot
# ===========================

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Advertising Platform")

plt.ylabel("Importance")

plt.title("Feature Importance")

plt.grid(True)

plt.show()

# ===========================
# 15. Save Model
# ===========================

joblib.dump(model, "sales_prediction_model.pkl")

print("\nModel Saved Successfully!")

# ===========================
# 16. Load Saved Model
# ===========================

loaded_model = joblib.load("sales_prediction_model.pkl")

print("Saved Model Loaded Successfully!")

# ===========================
# 17. Sample Prediction
# ===========================

sample = pd.DataFrame({
    "TV":[230.1],
    "Radio":[37.8],
    "Newspaper":[69.2]
})

prediction = loaded_model.predict(sample)

print("\nSample Prediction")
print("-----------------------")
print("Predicted Sales :", round(prediction[0],2))

# ===========================
# 18. User Input Prediction
# ===========================

print("\n===================================")
print("SALES PREDICTION SYSTEM")
print("===================================")

try:

    tv = float(input("Enter TV Advertising Budget : "))

    radio = float(input("Enter Radio Advertising Budget : "))

    newspaper = float(input("Enter Newspaper Advertising Budget : "))

    sample = pd.DataFrame({

        "TV":[tv],

        "Radio":[radio],

        "Newspaper":[newspaper]

    })

    sales = loaded_model.predict(sample)

    print("\nEstimated Sales")
    print("--------------------------")
    print(round(sales[0],2))

except Exception as e:

    print("Invalid Input")
    print(e)

# ===========================
# 19. Predict Multiple Campaigns
# ===========================

campaigns = pd.DataFrame({

    "TV":[150,250,300],

    "Radio":[20,35,45],

    "Newspaper":[15,40,60]

})

predictions = loaded_model.predict(campaigns)

print("\nMultiple Campaign Predictions")

for i in range(len(campaigns)):

    print(f"\nCampaign {i+1}")

    print("TV Budget :", campaigns.iloc[i]["TV"])

    print("Radio Budget :", campaigns.iloc[i]["Radio"])

    print("Newspaper Budget :", campaigns.iloc[i]["Newspaper"])

    print("Predicted Sales :", round(predictions[i],2))

# ===========================
# 20. Project Completed
# ===========================

print("\n" + "="*60)
print("SALES PREDICTION PROJECT COMPLETED SUCCESSFULLY")
print("="*60)