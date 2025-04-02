import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1.1: Load Data
train_data = pd.read_csv(r"C:\Users\eupho\OneDrive\Desktop\CODSOFT\Titanic-Dataset.csv")

# Step 1.2: Understand the Data
print("First few rows of the dataset:")
print(train_data.head())

print("\nDataset Info:")
train_data.info()

print("\nMissing Values in Each Column:")
print(train_data.isnull().sum())

# Step 1.3: Exploratory Data Analysis (EDA)

# 1. Survival Count
sns.countplot(x="Survived", data=train_data)
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
sns.countplot(x="Survived", hue="Sex", data=train_data)
plt.title("Survival by Gender")
plt.show()

# 3. Survival by Passenger Class
sns.countplot(x="Survived", hue="Pclass", data=train_data)
plt.title("Survival by Passenger Class")
plt.show()