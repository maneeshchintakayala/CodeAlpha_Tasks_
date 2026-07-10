# ============================================================
# IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING
# ============================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================================================
# STEP 1: Load Dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("="*60)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("="*60)

print("\nFeature Names:")
for feature in feature_names:
    print(feature)

print("\nTarget Classes:")
for target in target_names:
    print(target)

# ============================================================
# STEP 2: Create DataFrame
# ============================================================

df = pd.DataFrame(X, columns=feature_names)
df["Species"] = y

print("\nFirst Five Records")
print(df.head())

# ============================================================
# STEP 3: Dataset Information
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# STEP 4: Class Distribution
# ============================================================

print("\nClass Distribution")
print(df["Species"].value_counts())

# ============================================================
# STEP 5: Scatter Plot
# ============================================================

plt.figure(figsize=(8,6))

colors = ['red','green','blue']

for i, color in enumerate(colors):
    plt.scatter(
        X[y==i,0],
        X[y==i,2],
        color=color,
        label=target_names[i]
    )

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# STEP 6: Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape :", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ============================================================
# STEP 7: Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# STEP 8: Train Model
# ============================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ============================================================
# STEP 9: Predictions
# ============================================================

y_pred = model.predict(X_test)

print("\nPredicted Labels")
print(y_pred)

# ============================================================
# STEP 10: Model Accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score")
print(f"{accuracy*100:.2f}%")

# ============================================================
# STEP 11: Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ============================================================
# STEP 12: Classification Report
# ============================================================

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names
))

# ============================================================
# STEP 13: Feature Importance
# ============================================================

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ============================================================
# STEP 14: Feature Importance Plot
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=20)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.grid(True)

plt.show()

# ============================================================
# STEP 15: Save Model
# ============================================================

joblib.dump(model, "iris_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved Successfully!")

# ============================================================
# STEP 16: Load Saved Model
# ============================================================

loaded_model = joblib.load("iris_classifier.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("Saved Model Loaded Successfully!")

# ============================================================
# STEP 17: Predict Single Flower
# ============================================================

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = loaded_scaler.transform(new_flower)

prediction = loaded_model.predict(new_flower_scaled)

print("\nPrediction for Single Flower")
print("Flower Measurements :", new_flower)
print("Predicted Species :", target_names[prediction[0]])

# ============================================================
# STEP 18: Predict Multiple Flowers
# ============================================================

flowers = [
    [5.1,3.5,1.4,0.2],
    [6.0,2.9,4.5,1.5],
    [6.5,3.0,5.5,2.0],
    [4.9,3.0,1.4,0.2],
    [6.7,3.1,4.7,1.5]
]

flowers_scaled = loaded_scaler.transform(flowers)

predictions = loaded_model.predict(flowers_scaled)

print("\nPredictions for Multiple Flowers")

for i, flower in enumerate(flowers):
    print(f"\nFlower {i+1}")
    print("Measurements :", flower)
    print("Predicted Species :", target_names[predictions[i]])

# ============================================================
# STEP 19: Predict Custom User Input
# ============================================================

print("\n==============================")
print("CUSTOM FLOWER PREDICTION")
print("==============================")

try:
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    sample_scaled = loaded_scaler.transform(sample)

    prediction = loaded_model.predict(sample_scaled)

    print("\nPredicted Flower Species:")
    print(target_names[prediction[0]])

except:
    print("Invalid Input!")

# ============================================================
# STEP 20: Project Completed
# ============================================================

print("\n" + "="*60)
print("IRIS FLOWER CLASSIFICATION PROJECT COMPLETED SUCCESSFULLY")
print("="*60)