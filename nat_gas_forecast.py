
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the CSV file
file_path = 'Nat_Gas.csv'
nat_gas_data = pd.read_csv(file_path)

# Convert the 'Dates' column to datetime format
nat_gas_data['Dates'] = pd.to_datetime(nat_gas_data['Dates'], format='%m/%d/%y')

# Set 'Dates' as the index for better time series analysis
nat_gas_data.set_index('Dates', inplace=True)

# Fit the Holt-Winters model to the data
hw_model = ExponentialSmoothing(nat_gas_data['Prices'], seasonal='add', seasonal_periods=12, trend='add').fit()

# Forecast for the next 12 months
hw_forecast = hw_model.forecast(steps=12)

# Create a DataFrame to store the forecasted results
hw_forecast_dates = pd.date_range(start=nat_gas_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
hw_forecast_df = pd.DataFrame({'Forecasted_Prices': hw_forecast}, index=hw_forecast_dates)

# Combine the original data with the forecasted data
combined_hw_df = pd.concat([nat_gas_data, hw_forecast_df])

# Plot the original data along with the forecast
plt.figure(figsize=(12, 6))
plt.plot(combined_hw_df.index, combined_hw_df['Prices'], label='Observed Prices', marker='o')
plt.plot(hw_forecast_df.index, hw_forecast_df['Forecasted_Prices'], label='Forecasted Prices', marker='x')
plt.title('Natural Gas Prices Forecast using Holt-Winters Exponential Smoothing')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Output the forecasted prices
print(hw_forecast_df)
