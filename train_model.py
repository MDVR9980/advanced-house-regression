import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test = joblib.load('prepared_data.pkl')

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
