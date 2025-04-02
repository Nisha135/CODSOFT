import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Perform basic preprocessing (handle missing values, encode categorical data, etc.)
data.fillna(method='ffill', inplace=True)

data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Select relevant features and target variable
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

X = data[features]
y = data[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed dataset
preprocessed_data = pd.concat([X, y], axis=1)
preprocessed_data.to_csv("titanic_preprocessed.csv", index=False)

# Save training and testing data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing complete. Files saved successfully.")