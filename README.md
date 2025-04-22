# 🏡 Advanced House Regression

This project focuses on building advanced regression models to predict house prices using machine learning techniques. The dataset used is based on the popular Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), with the dataset sourced from Hands-On ML by Aurélien Géron.

## 🚀 Overview

The goal is to predict the final sale price of each house based on a rich set of features. We explore various regression models and preprocessing steps to achieve high accuracy.

### 🔍 Features

- Data cleaning and preprocessing (handling missing values, encoding categorical features, feature engineering)
- Feature scaling using `StandardScaler`
- Model training and evaluation using `LinearRegression`
- Batch prediction from CSV files and export of predicted results
- Interactive UI with `Streamlit` for real-time predictions

---

## 📁 Project Structure

```
advanced-house-regression/
├── housing.csv                  # Dataset file
├── data_preprocessing.py       # Preprocesses and saves the training data
├── train_model.py              # Trains the model and saves the scaler/model
├── predict.py                  # Script to make a single prediction
├── batch_predict.py            # Predicts house prices from new_data.csv and exports predictions.csv
├── app.py                      # Streamlit app for web-based prediction
├── new_data.csv                # Sample new data for batch predictions
├── predictions.csv             # Output CSV with predictions
├── scaler.pkl                  # Saved StandardScaler object
├── linear_regression_model.pkl # Trained model file
├── prepared_data.pkl           # Scaled and split dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📅 Setup & Usage

### ⚡ Installation

```bash
pip install -r requirements.txt
```

### 🎮 Run Preprocessing and Training

```bash
python3 data_preprocessing.py
python3 train_model.py
```

### 💰 Predict Single Input (CLI)

```bash
python3 predict.py
```

### 📤 Batch Predict from CSV

```bash
python3 batch_predict.py
```

### 🌐 Run Streamlit App

```bash
streamlit run app.py
```

---

## 📄 Dataset

Dataset used: [housing.csv](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)

---

## 📃 Requirements

```
pandas
scikit-learn
joblib
streamlit
```

---

## 📈 Example Output

```
Predicted House Price: $415,721
Predictions saved to predictions.csv
```



