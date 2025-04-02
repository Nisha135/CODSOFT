# Titanic Survival Prediction

## 🚢 Project Overview
The **Titanic Survival Prediction** project is a machine learning model that predicts whether a passenger would survive the Titanic disaster based on various features like age, gender, passenger class, and more. The model is trained using the Titanic dataset from Kaggle.

## 📂 Dataset
The dataset used in this project is the famous Titanic dataset, which contains information about passengers, including:
- **Passenger ID**
- **Survived** (Target variable: 0 = No, 1 = Yes)
- **Pclass** (Ticket class: 1st, 2nd, 3rd)
- **Name, Sex, Age**
- **SibSp** (Number of siblings/spouses aboard)
- **Parch** (Number of parents/children aboard)
- **Ticket, Fare, Cabin, Embarked** (Port of Embarkation)

## 🔧 Installation
To run this project locally, follow these steps:

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Nisha135/CODSOFT.git
cd CODSOFT
```

### 2️⃣ Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

## 🚀 Usage
### 1️⃣ Train the Model
Run the script to train the model:
```sh
python train.py
```

### 2️⃣ Make Predictions
Once trained, you can use the model to make predictions:
```sh
python predict.py --input sample_data.csv
```

## 🌐 Deployment
The model is deployed as a **Flask API**. To start the API server, run:
```sh
python app.py
```
Access the API at: `http://127.0.0.1:5000/predict`

## 📜 License
This project is open-source and available under the **MIT License**.

---

Made with ❤️ by **Nisha Ravi**

