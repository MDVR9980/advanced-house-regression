import joblib
import pandas as pd

# Load the scaler and trained model
scaler = joblib.load('scaler.pkl')
model = joblib.load('linear_regression_model.pkl')

# Example new data for prediction
new_data = {
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
}

# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# One-hot encode 'ocean_proximity'
new_df = pd.get_dummies(new_df, columns=['ocean_proximity'], drop_first=True)

# Load training data columns (from processed training data)
# We extract the columns used during training to match them exactly
X_train, _, _, _ = joblib.load('prepared_data.pkl')
original_columns = pd.get_dummies(
    pd.read_csv('housing.csv')
    .drop('median_house_value', axis=1)
    .select_dtypes(include=['number', 'object']),
    columns=['ocean_proximity'],
    drop_first=True
).columns

# Add any missing columns in new_df with 0s
for col in original_columns:
    if col not in new_df.columns:
        new_df[col] = 0

# Ensure column order matches training data
new_df = new_df[original_columns]

# Scale the data
new_data_scaled = scaler.transform(new_df)

# Predict
predicted_price = model.predict(new_data_scaled)

print(f"Predicted House Price: {predicted_price[0]}")
