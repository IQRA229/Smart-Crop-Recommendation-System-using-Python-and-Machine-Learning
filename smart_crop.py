# crop_recommendation.py
# Smart Crop Recommendation System by Iqra

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

# Step 3: Prepare features (X) and target (y)
X = df[['Soil_Type', 'Rainfall', 'Temperature', 'pH']]
y = df['Recommended_Crop']

# Step 4: Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 5: Default input values
print("🌾 Welcome to Smart Crop Recommendation System 🌿")
print("Using predefined input values for prediction...\n")

# Default values (change these as needed)
soil = 1        # 1=Loamy
rain = 210.0    # mm
temp = 27.0     # °C
ph = 6.5        # pH

# You can also use user input by uncommenting the following:
# soil = int(input("Enter Soil Type (1=Loamy, 2=Sandy, 3=Clay): "))
# rain = float(input("Enter average Rainfall (in mm): "))
# temp = float(input("Enter average Temperature (°C): "))
# ph = float(input("Enter Soil pH: "))

# Step 6: Make prediction
prediction = model.predict([[soil, rain, temp, ph]])

# Step 7: Show result
print("Based on the data entered:")
print(f"   Soil Type: {soil}, Rainfall: {rain} mm, Temperature: {temp}°C, pH: {ph}")
print("Recommended Crop for You:", prediction[0])
