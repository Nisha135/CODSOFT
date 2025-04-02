import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
data = pd.read_csv("titanic_preprocessed.csv")  

# Splitting features and labels
X = data.drop("Survived", axis=1)
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for future use
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)  # Saving y_test as well

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Save trained model
import joblib
joblib.dump(model, "titanic_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy : {accuracy:.2f}")
