# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.
## Problem Statement and Dataset

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Collect and Prepare Data:Gather employee data such as experience, age, education, and position. Clean the dataset and split it into input features (X) and target salary values (Y).
2.Split the Dataset:Divide the dataset into training data and testing data using train-test split methods.
3.Train the Decision Tree Regressor Model:Import the Decision Tree Regressor algorithm, create the model, and train it using the training dataset.
4.Predict and Evaluate Salary:Use the trained model to predict employee salaries for test data and evaluate the model performance using metrics like Mean Absolute Error (MAE) or R² score.
## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by:MENAKA M S
RegisterNumber: 212225040232
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Create Dataset
data = {
    "Temperature": [30, 32, 35, 28, 25, 27, 33, 31, 29, 26],
    "Humidity": [70, 65, 60, 80, 90, 85, 55, 68, 75, 88],
    "WindSpeed": [10, 12, 8, 15, 20, 18, 7, 11, 14, 19],
    "Weather": [
        "Sunny", "Sunny", "Sunny", "Rainy", "Rainy",
        "Rainy", "Sunny", "Sunny", "Rainy", "Rainy"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display Dataset
print("Dataset:\n")
print(df)

# Step 3: Convert Categorical Labels to Numeric
df["Weather"] = df["Weather"].map({
    "Sunny": 1,
    "Rainy": 0
})

# Features and Target
X = df[["Temperature", "Humidity", "WindSpeed"]]
y = df["Weather"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Evaluation:")

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

# Step 9: Visualization of Feature Importance
plt.figure(figsize=(8,6))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")

plt.grid(True)

plt.show()

# Step 10: Custom Prediction
weather_data = [[29, 72, 13]]

prediction = model.predict(weather_data)

print("\nCustom Weather Prediction:")

if prediction[0] == 1:
    print("Predicted Weather: Sunny")
else:
    print("Predicted Weather: Rainy")

```

## Output:
<img width="481" height="363" alt="image" src="https://github.com/user-attachments/assets/70546dd5-8d1a-44a9-96d9-4b7f0022967f" />
<img width="672" height="760" alt="image" src="https://github.com/user-attachments/assets/4fa44528-b95f-4757-bfec-1cc23500bf89" />
<img width="250" height="57" alt="image" src="https://github.com/user-attachments/assets/17f15a55-b41d-46f6-8261-5f9882f9450f" />
## Result:
Succesfully,the program is to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.
