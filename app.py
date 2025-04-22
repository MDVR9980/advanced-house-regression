import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Get original columns
X_train, _, _, _ = joblib.load('prepared_data.pkl')
original_columns = pd.get_dummies(
    pd.read_csv('housing.csv')
    .drop('median_house_value', axis=1),
    columns=['ocean_proximity'],
    drop_first=True
).columns

st.title("🏠 House Price Predictor")

# Form inputs
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

if st.button("Predict"):
    input_dict = {
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    }

    input_df = pd.DataFrame(input_dict)
    input_df = pd.get_dummies(input_df, columns=['ocean_proximity'], drop_first=True)

    for col in original_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[original_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"🏷️ Estimated House Price: ${prediction[0]:,.0f}")
