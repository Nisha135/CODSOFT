import pandas as pd
import joblib  # For loading the saved model

# Load the trained model
model = joblib.load("titanic_model.pkl")  # 🔥 Loads the model saved in step4.py

# Load the test dataset
test_data = pd.read_csv("Titanic-Dataset.csv")

# Select features (same as in training)
X_test = test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked'], drop_first=True)

# Make predictions
y_pred = model.predict(X_test)

# Display the first few predictions
print("Predictions on test data:")
print(y_pred[:10])  # Show first 10 predictions
