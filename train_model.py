import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load('prepared_data.pkl')

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(model, 'linear_regression_model.pkl')  # ⬅️ این خط اضافه شد
