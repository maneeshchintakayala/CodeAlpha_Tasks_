# ==============================================================
# CAR PRICE PREDICTION USING MACHINE LEARNING
# Dataset: car data.csv
# ==============================================================

# ==========================
# 1. Import Libraries
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ==========================
# 2. Load Dataset
# ==========================

df = pd.read_csv("car data.csv")

print("="*60)
print("CAR PRICE PREDICTION PROJECT")
print("="*60)

print("\nFirst Five Records")
print(df.head())

# ==========================
# 3. Dataset Information
# ==========================

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================
# 4. Feature Engineering
# ==========================

current_year = 2025

df["Car_Age"] = current_year - df["Year"]

df.drop("Year", axis=1, inplace=True)

# ==========================
# 5. Encode Categorical Columns
# ==========================

label_encoders = {}

categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("\nEncoded Dataset")
print(df.head())

# ==========================
# 6. Define Features & Target
# ==========================

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

print("\nFeatures")
print(X.columns)

# ==========================
# 7. Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

# ==========================
# 8. Train Random Forest Model
# ==========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# 9. Predictions
# ==========================

y_pred = model.predict(X_test)

print("\nPredicted Prices")
print(y_pred[:10])

# ==========================
# 10. Model Evaluation
# ==========================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score :", r2)

# ==========================
# 11. Feature Importance
# ==========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ==========================
# 12. Plot Feature Importance
# ==========================

plt.figure(figsize=(10,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.xlabel("Features")

plt.ylabel("Importance")

plt.title("Feature Importance")

plt.grid(True)

plt.show()

# ==========================
# 13. Actual vs Predicted
# ==========================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Selling Price")

plt.grid(True)

plt.show()

# ==========================
# 14. Save Model
# ==========================

joblib.dump(model, "car_price_model.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel Saved Successfully!")

# ==========================
# 15. Load Saved Model
# ==========================

loaded_model = joblib.load("car_price_model.pkl")

loaded_encoders = joblib.load("label_encoders.pkl")

print("Saved Model Loaded Successfully!")

# ==========================
# 16. Predict Sample Car
# ==========================

sample = pd.DataFrame({

    "Car_Name":[loaded_encoders["Car_Name"].transform(["ritz"])[0]],

    "Present_Price":[5.59],

    "Driven_kms":[27000],

    "Fuel_Type":[loaded_encoders["Fuel_Type"].transform(["Petrol"])[0]],

    "Selling_type":[loaded_encoders["Selling_type"].transform(["Dealer"])[0]],

    "Transmission":[loaded_encoders["Transmission"].transform(["Manual"])[0]],

    "Owner":[0],

    "Car_Age":[11]

})

prediction = loaded_model.predict(sample)

print("\nSample Car Prediction")
print("------------------------")
print("Predicted Selling Price :", round(prediction[0],2), "Lakhs")

# ==========================
# 17. User Input Prediction
# ==========================

print("\n==============================")
print("CAR PRICE PREDICTION SYSTEM")
print("==============================")

try:

    car_name = input("Enter Car Name : ")

    present_price = float(input("Present Price (Lakhs): "))

    driven = int(input("Driven Kilometers : "))

    fuel = input("Fuel Type (Petrol/Diesel/CNG): ")

    selling_type = input("Selling Type (Dealer/Individual): ")

    transmission = input("Transmission (Manual/Automatic): ")

    owner = int(input("Owner Count : "))

    year = int(input("Manufacturing Year : "))

    age = current_year - year

    if car_name not in loaded_encoders["Car_Name"].classes_:
        print("\nCar name not found in dataset.")
    else:

        sample = pd.DataFrame({

            "Car_Name":[loaded_encoders["Car_Name"].transform([car_name])[0]],

            "Present_Price":[present_price],

            "Driven_kms":[driven],

            "Fuel_Type":[loaded_encoders["Fuel_Type"].transform([fuel])[0]],

            "Selling_type":[loaded_encoders["Selling_type"].transform([selling_type])[0]],

            "Transmission":[loaded_encoders["Transmission"].transform([transmission])[0]],

            "Owner":[owner],

            "Car_Age":[age]

        })

        price = loaded_model.predict(sample)

        print("\nEstimated Selling Price")
        print("----------------------------")
        print("₹", round(price[0],2), "Lakhs")

except Exception as e:
    print("\nInvalid Input!")
    print(e)

# ==========================
# 18. Project Completed
# ==========================

print("\n"+"="*60)
print("CAR PRICE PREDICTION PROJECT COMPLETED SUCCESSFULLY")
print("="*60)