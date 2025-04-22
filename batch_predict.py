import joblib
import pandas as pd

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('linear_regression_model.pkl')

# Load new data
new_df = pd.read_csv('new_data.csv')

# One-hot encoding
new_df = pd.get_dummies(new_df, columns=['ocean_proximity'], drop_first=True)

# Get training columns for alignment
X_train, _, _, _ = joblib.load('prepared_data.pkl')
original_columns = pd.get_dummies(
    pd.read_csv('housing.csv')
    .drop('median_house_value', axis=1),
    columns=['ocean_proximity'],
    drop_first=True
).columns

# Add missing columns
for col in original_columns:
    if col not in new_df.columns:
        new_df[col] = 0
new_df = new_df[original_columns]

# Scale and predict
scaled = scaler.transform(new_df)
predictions = model.predict(scaled)

# Save results
new_df['predicted_price'] = predictions
new_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
