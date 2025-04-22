import joblib
import pandas as pd

# Load the scaler, model, and label encoder
scaler = joblib.load('scaler.pkl')
model = joblib.load('linear_regression_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Example new data for prediction (you can replace this with actual new data)
new_data = {
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']  # Categorical data
}

# Convert new data to DataFrame
new_df = pd.DataFrame(new_data)

# Step 1: Convert categorical column 'ocean_proximity' to numeric
new_df['ocean_proximity'] = label_encoder.transform(new_df['ocean_proximity'])

# Step 2: Scale the features using the loaded scaler
new_data_scaled = scaler.transform(new_df)

# Step 3: Predict the price using the trained model
predicted_price = model.predict(new_data_scaled)

print(f"Predicted House Price: {predicted_price[0]}")
