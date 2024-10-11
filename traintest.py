import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('stock_price.csv')

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Rename columns to match Prophet's requirements
df.rename(columns={'Date': 'ds', 'closing-price': 'y'}, inplace=True)

# Convert 'ds' to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Ensure 'y' is treated as a string, then remove commas and 'M'
df['y'] = df['y'].astype(str).str.replace('M', '').str.replace(',', '')

# Convert 'y' to numeric
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Drop rows with missing values
df.dropna(subset=['ds', 'y'], inplace=True)

# Split the data into training and test sets
train_size = int(len(df) * 0.85)
train_df = df[:train_size]
test_df = df[train_size:]

# Initialize the model
model = Prophet()

# Fit the model on the training set
model.fit(train_df)

# Create a DataFrame for future dates in the test set
future = model.make_future_dataframe(periods=len(test_df), include_history=False)

# Make predictions
forecast = model.predict(future)

# Evaluate the accuracy
y_true = test_df['y'].values
y_pred = forecast['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')

# Print the first few rows of the actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
print(comparison_df.head())

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_df['ds'], y_true, label='Actual')
plt.plot(test_df['ds'], y_pred, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices')
plt.legend()
plt.show()