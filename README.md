# SMART_SOIL: AI-powered Soil Moisture and Crop Type Detection

## 📌 Project Overview
SMART_SOIL is a machine learning-based analysis model for predicting **soil moisture levels** and **crop type classification** based on satellite data and geo-location. This project leverages **Sentinel-1 and Sentinel-2 satellite imagery** to provide insights that can aid farmers in precise farming.

## 🚀 Features
- 🌱 **Soil Moisture Prediction** using Sentinel-1 SAR bands (VH, VV)
- 🌾 **Crop Type Classification** based on Sentinel-2 spectral bands (B4, B8, B11, B12)
- 📍 **Geo-Location Based Insights** by integrating location coordinates
- 📊 **Data Visualization & EDA** for better understanding of soil and crop conditions

## 📂 Dataset
**Name:** SMART_SOIL: AI-powered Soil Moisture and Crop Type Detection  
**Source:** Satellite data from Sentinel-1 & Sentinel-2  
**Columns:**
- `B11, B12, B4, B8` → Spectral bands from Sentinel-2
- `VH, VV` → SAR backscatter bands from Sentinel-1
- `.geo` → Geo-coordinates (to be extracted as latitude & longitude)
- **Missing:** Crop type labels (required for classification model)

## 🔧 Tech Stack
- **Programming Language:** Python 🐍
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, XGBoost
- **ML Models:** Random Forest, XGBoost, Decision Trees

## 📊 Project Workflow
1. **Data Preprocessing:** Extract latitude & longitude, clean data
2. **Feature Engineering:** Compute NDVI, NDWI for better predictions
3. **Exploratory Data Analysis (EDA):** Identify correlations and visualize data
4. **Model Training:** Train models for soil moisture prediction and crop classification
5. **Evaluation:** Measure accuracy using RMSE (for regression) and F1-score (for classification)

## ❗ Challenges & Limitations
- 🔹 **No Direct Soil Moisture Labels** → Needs proxy indicators like NDWI
- 🔹 **Missing Crop Labels** → Can’t train classification model without labeled data
- 🔹 **Need External Validation** → Accuracy depends on real-world data availability

## 🛠️ Future Improvements
✅ Integrate real-world soil moisture and crop datasets  
✅ Build a web-based tool for farmer accessibility  
✅ Improve accuracy with deep learning models (CNNs for classification)  

## 📌 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SMART_SOIL.git
   cd SMART_SOIL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing & model scripts:
   ```bash
   python preprocess.py
   python train_model.py
   ```

## 📝 Authors & Credits
Developed by Swayam and Team  
Contributions & feedback are welcome! 🚀

