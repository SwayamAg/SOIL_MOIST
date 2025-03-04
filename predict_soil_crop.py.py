# WILL DO SOME CHANGES

import numpy as np
import pandas as pd
import joblib

# Load models & encoders
soil_moisture_model = joblib.load("soil_moisture_model.pkl")
crop_classification_model = joblib.load("crop_classification_model.pkl")
scaler_soil = joblib.load("scaler.pkl")  # The original scaler trained on soil features
label_encoder = joblib.load("label_encoder.pkl")

# Define feature names
soil_features = ["B11", "B12", "B4", "B8", "VH", "VV", "NDVI"]
crop_features = ["B11", "B12", "B4", "B8", "VH", "VV", "NDVI", "NDWI"]

# Generate a random test input for soil moisture prediction (7 features)
X_soil_test = np.random.rand(1, 7)

# Convert to DataFrame for proper scaling
X_soil_df = pd.DataFrame(X_soil_test, columns=soil_features)

# Scale only soil moisture features
X_soil_scaled = scaler_soil.transform(X_soil_df)

# Predict soil moisture (NDWI)
ndwi_prediction = soil_moisture_model.predict(X_soil_scaled)[0]

# Print Soil Moisture Prediction
print(f"🌱 Soil Moisture Prediction (NDWI): {ndwi_prediction:.4f}")

# Append NDWI to the crop classification input
X_crop_test = np.append(X_soil_test, ndwi_prediction).reshape(1, -1)

# Since scaler was trained without NDWI, we manually normalize it between 0 and 1
ndwi_min, ndwi_max = -1, 1  # Assuming NDWI was scaled in this range
ndwi_scaled = (ndwi_prediction - ndwi_min) / (ndwi_max - ndwi_min)

# Create DataFrame for crop classification
X_crop_df = pd.DataFrame(X_crop_test, columns=crop_features)

# Scale only the first 7 features (excluding NDWI)
X_crop_df.iloc[:, :-1] = scaler_soil.transform(X_crop_df.iloc[:, :-1])

# Manually add the scaled NDWI
X_crop_df["NDWI"] = ndwi_scaled

# **Fix Warning:** Convert DataFrame to NumPy array before prediction
X_crop_np = X_crop_df.values

# Predict crop type
crop_type_encoded = crop_classification_model.predict(X_crop_np)[0]

# Decode label to crop name
crop_type = label_encoder.inverse_transform([crop_type_encoded])[0]

# Print Recommended Crop Type
print(f"🌾 Recommended Crop Type: {crop_type}")
