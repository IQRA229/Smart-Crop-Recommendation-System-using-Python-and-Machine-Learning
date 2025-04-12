# crop_recommendation_rf.py
# Smart Crop Recommendation System using Random Forest - by Iqra

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create sample dataset
data = {
    'Soil_Type': [1, 2, 3, 1, 2, 3, 1, 2],
    'Rainfall': [200, 150, 300, 250, 100, 280, 220, 180],
    'Temperature': [25, 20, 30, 28, 22, 31, 26, 24],
    'pH': [6.5, 7.0, 5.8, 6.8, 7.2, 6.0, 6.7, 7.1],
    'Recommended_Crop': ['Wheat', 'Barley', 'Rice', 'Maize', 'Wheat', 'Rice', 'Maize', 'Barley']
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# Step 3: Prepare features and labels
X = df[['Soil_Type', 'Rainfall', 'Temperature', 'pH']]
y = df['Recommended_Crop']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 7: Welcome message
print("🌾 Welcome to Smart Crop Recommendation System 🌿")
print("Using predefined input values for prediction...\n")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Step 8: Default input values
soil = 1        # Loamy
rain = 210.0    # mm
temp = 27.0     # °C
ph = 6.5        # pH

# Step 9: Prepare input
input_data = pd.DataFrame(np.array([[soil, rain, temp, ph]]), columns=['Soil_Type', 'Rainfall', 'Temperature', 'pH'])

# Step 10: Make prediction
prediction = model.predict(input_data)

# Step 11: Show result
print("Based on the data entered:")
print(f"   Soil Type: {soil}, Rainfall: {rain} mm, Temperature: {temp}°C, pH: {ph}")
print("🌱 Recommended Crop for You:", prediction[0])
