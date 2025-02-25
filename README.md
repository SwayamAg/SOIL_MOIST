# 🌾 Soil Moisture and Crop Prediction using Satellite Data

## 📌 Project Overview
This project leverages satellite data (Sentinel-1 and Sentinel-2) to predict **soil moisture (NDWI)** and classify **crop types** using machine learning techniques. The goal is to assist farmers in **Uttar Pradesh, India** by providing insights on soil fertility and crop recommendations.

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Machine Learning** (Scikit-learn, XGBoost, Random Forest)
- **Google Earth Engine (GEE)** (Data Processing for Sentinel-1 & Sentinel-2)
- **Joblib** (Model Serialization)
- **Jupyter Notebook / PyCharm** (Development Environment)

## 📊 Dataset Details
- **Source**: Sentinel-1 & Sentinel-2 processed using **Google Earth Engine (GEE)**
- **Key Features**:
  - **Spectral Bands**: B4, B8, B11, B12 (from Sentinel-2), VH, VV (from Sentinel-1)
  - **Indices**: NDVI (Normalized Difference Vegetation Index), NDWI (Normalized Difference Water Index)
  - **Latitude & Longitude**
  - **Crop Labels**: Millets, Wheat, Barley, Sugarcane, Rice, Jute

## 📌 Model Overview
### 1️⃣ **Soil Moisture Prediction (Regression Model)**
- **Algorithm**: XGBoost Regressor
- **Input Features**: Sentinel-1 & Sentinel-2 bands, NDVI, VH, VV
- **Target**: NDWI (Soil Moisture Indicator)
- **Performance**:
  - **Mean Squared Error (MSE):** 0.0001
  - **R-squared Score:** 0.9675

### 2️⃣ **Crop Type Classification (Classification Model)**
- **Algorithm**: Random Forest Classifier
- **Features Used**: Sentinel-1 & Sentinel-2 bands, NDVI, NDWI
- **Class Balancing**: Used `class_weight="balanced"`
- **Performance**:
  - **Accuracy:** 99.9%
  - **Precision:** 99.9%
  - **Recall:** 99.9%
  - **F1 Score:** 99.9%

## 📁 Project Structure
```
├── data/                    # Dataset (CSV format, processed via GEE)
├── models/                  # Trained Models (saved using Joblib)
├── notebooks/               # Jupyter Notebooks for analysis & visualization
├── scripts/                 # Python scripts for training & evaluation
├── README.md                # Project Documentation
└── requirements.txt         # Dependencies & Libraries
```

## 📈 Results & Visualizations
- **Heatmap** of feature correlations.
- **Feature Importance** plot for classification.
- **Comparison of actual vs predicted NDWI values**.

## 🚀 How to Run the Project
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/soil-moisture-crop-prediction.git
   cd soil-moisture-crop-prediction
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Model Training Script**:
   ```sh
   python scripts/train_models.py
   ```

4. **Evaluate the Model**:
   ```sh
   python scripts/evaluate_models.py
   ```

## 📌 Future Improvements
- Fine-tune hyperparameters for even better accuracy.
- Expand dataset to include more regions.
- Deploy a web-based dashboard for farmers.
- Integrate real-time satellite data updates.

## 💡 Contributors
- **Swayam Agarwal** - Developer & Researcher

## 📜 License
This project is licensed under the **MIT License**.

---
🚀 **Empowering Agriculture with AI & Satellite Data!** 🌍

