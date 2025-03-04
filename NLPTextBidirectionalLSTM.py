import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Sample text data and labels (for illustration)
texts = [
    "I love deep learning",
    "Sequence models are fascinating",
    "Natural language processing is powerful",
    "Recurrent neural networks capture context"
]
labels = [1, 1, 1, 0]  # Example binary labels

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(labels)

# Build the Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model (using a small number of epochs for demonstration)
model.fit(X, y, epochs=5, batch_size=2)
