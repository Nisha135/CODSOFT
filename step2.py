# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset (Ensure the path is correct)
train_data = pd.read_csv(r"C:\Users\eupho\OneDrive\Desktop\CODSOFT\Titanic-Dataset.csv")

# Display first few rows
print("First few rows of the dataset:")
print(train_data.head())

# Display dataset information
train_data.info()

# Check for missing values
print("\nMissing Values in Each Column:\n")
print(train_data.isnull().sum())

# Handle missing values
train_data["Age"].fillna(train_data["Age"].median(), inplace=True)  # Fill missing Age with median
train_data["Embarked"].fillna(train_data["Embarked"].mode()[0], inplace=True)  # Fill missing Embarked with most common value

# Drop the 'Cabin' column (too many missing values)
train_data.drop(columns=['Cabin'], inplace=True)

# Encode categorical variables (Sex & Embarked)
label_encoder = LabelEncoder()
train_data["Sex"] = label_encoder.fit_transform(train_data["Sex"])  # Convert 'Sex' (male=1, female=0)
train_data["Embarked"] = label_encoder.fit_transform(train_data["Embarked"])  # Convert 'Embarked' to numbers

# Scale numerical features (Fare & Age)
scaler = StandardScaler()
train_data[["Fare", "Age"]] = scaler.fit_transform(train_data[["Fare", "Age"]])

# Display first few rows after preprocessing
print("\nDataset after preprocessing:\n")
print(train_data.head())

# Data Visualization
sns.countplot(x="Survived", data=train_data)
plt.title("Survival Count")
plt.show()

sns.countplot(x="Survived", hue="Sex", data=train_data)
plt.title("Survival by Gender")
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=train_data)
plt.title("Survival by Passenger Class")
plt.show()
