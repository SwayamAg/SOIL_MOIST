import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'UP_Soil_Moisture_Crop_Data.csv'
df = pd.read_csv(file_path)

# Preview the first few rows of the dataset
print("First few rows:")
print(df.head())
print(df.columns)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Get basic statistics
print("\nBasic statistics:")
print(df.describe())

# Display data types
print("\nData types:")
print(df.dtypes)

