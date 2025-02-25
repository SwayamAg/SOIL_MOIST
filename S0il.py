import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
file_path = "UP_Soil_Moisture_Crop_Data.csv"
df = pd.read_csv(file_path)

# Extract latitude & longitude from '.geo' column
def extract_coordinates(geo_str):
    geo_json = json.loads(geo_str)
    return pd.Series(geo_json["coordinates"])

df[["Longitude", "Latitude"]] = df[".geo"].apply(extract_coordinates)
df.drop(columns=[".geo", "system:index"], inplace=True)

# Compute NDVI and NDWI safely
def safe_index(numerator, denominator):
    return np.where(denominator == 0, 0, numerator / denominator)

df["NDVI"] = safe_index(df["B8"] - df["B4"], df["B8"] + df["B4"])
df["NDWI"] = safe_index(df["B8"] - df["B11"], df["B8"] + df["B11"])

# Improved Crop Classification Based on Domain Knowledge
def classify_crop(ndvi, ndwi):
    if ndwi < 0.05 and ndvi < 0.2:
        return "Millets"
    elif 0.05 <= ndwi < 0.3 and 0.2 <= ndvi < 0.5:
        return "Wheat"
    elif ndvi >= 0.5 and ndwi >= 0.3:
        return "Sugarcane"
    elif ndvi >= 0.5 and ndwi < 0.3:
        return "Rice"
    elif ndvi < 0.2 and ndwi >= 0.3:
        return "Jute"
    else:
        return "Barley"

df["CropType"] = df.apply(lambda row: classify_crop(row["NDVI"], row["NDWI"]), axis=1)

# Data visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# Soil Moisture Prediction (Regression Model)
X_reg = df[["B11", "B12", "B4", "B8", "VH", "VV", "NDVI"]]
y_reg = df["NDWI"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Train XGBoost Regressor
xgb_reg = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train_reg_scaled, y_train_reg)
joblib.dump(xgb_reg, "soil_moisture_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Evaluate Regression Model
y_pred_reg = xgb_reg.predict(X_test_reg_scaled)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Crop Type Classification (Classification Model)
X_cls = df[["B11", "B12", "B4", "B8", "VH", "VV", "NDVI", "NDWI"]]
y_cls = df["CropType"]

label_encoder = LabelEncoder()
y_cls_encoded = label_encoder.fit_transform(y_cls)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls_encoded, test_size=0.2, random_state=42)
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

# Use Class Weights Instead of SMOTE
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
rf_cls.fit(X_train_cls_scaled, y_train_cls)
joblib.dump(rf_cls, "crop_classification_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Evaluate Classification Model
y_pred_cls = rf_cls.predict(X_test_cls_scaled)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(y_test_cls, y_pred_cls, average='weighted', zero_division=1)
recall = recall_score(y_test_cls, y_pred_cls, average='weighted', zero_division=1)
f1 = f1_score(y_test_cls, y_pred_cls, average='weighted', zero_division=1)
print(f"Crop Classification Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls, target_names=label_encoder.classes_, zero_division=1))

# Feature Importance Plot
feature_importances = rf_cls.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importances, y=X_cls.columns)
plt.title("Feature Importance in Crop Classification Model")
plt.show()

# Compare Predictions
print("Comparing Soil Moisture Predictions (First 5 Samples):")
comparison_df = pd.DataFrame({
    "Actual NDWI": y_test_reg.values[:5],
    "Predicted NDWI": y_pred_reg[:5]
})
print(comparison_df)
