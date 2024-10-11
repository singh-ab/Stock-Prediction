import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('stock_price.csv')

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names to verify
print(df.columns)

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

# Initialize the model
model = Prophet()

# Fit the model
model.fit(df)

# Create a DataFrame for future dates
future = model.make_future_dataframe(periods=700)

# Predict future values
forecast = model.predict(future)

# Plot the forecast
fig1 = model.plot(forecast)
plt.show()

# Plot the seasonal components
fig2 = model.plot_components(forecast)
plt.show()