import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate synthetic time series data: sine wave with noise
time = np.arange(0, 100, 0.1)
series = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

# Create dataset using a sliding window approach
def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

window_size = 20
X, y = create_dataset(series, window_size)
X = X.reshape(-1, window_size, 1)

# Build the LSTM model for forecasting
model = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Plot predictions vs. true values for the last 100 data points
predicted = model.predict(X[-100:])
plt.plot(y[-100:], label='True')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()
