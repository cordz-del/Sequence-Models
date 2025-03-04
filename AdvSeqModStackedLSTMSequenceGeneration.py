import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Parameters for synthetic sequence data
vocab_size = 5000  # Example vocabulary size
max_length = 50    # Maximum sequence length

# Generate synthetic data: sequences of integers representing words
X = np.random.randint(1, vocab_size, size=(1000, max_length))
y = np.random.randint(1, vocab_size, size=(1000, max_length))  # Target sequences

# Build the stacked LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(256, return_sequences=True),
    LSTM(256, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Train the model (using a small number of epochs for demonstration)
model.fit(X, y, epochs=5, batch_size=32)
