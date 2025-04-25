import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Step 1: Load Stock Data
ticker = 'AAPL'  # You can change this to any stock ticker
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
print(df.head())

# Step 2: Visualization
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price')
plt.title(f'{ticker} Stock Price History')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.show()

# Step 3: Preprocessing
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 60
x_train = []
y_train = []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 4: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=64, epochs=10)

# Step 5: Testing and Predictions
test_data = scaled_data[-(sequence_length + 30):]  # last part of data
x_test = []
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Step 6: Plot the Predictions
actual_prices = data[-30:]
predicted_prices = predictions

plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual Price')
plt.plot(df.index[-30:], predicted_prices, color='red', label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.legend()
plt.show()

