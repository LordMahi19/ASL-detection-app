import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset you created
data = np.load("sign_language_dataset.npz")
X_train = data['data']
y_train = data['labels']

# Reshape the data for the model (assuming 42 features per hand)
X_train = X_train.reshape(X_train.shape[0], 42)

# One-hot encode the labels (if necessary)
# You might need to adjust this based on the format of your labels
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train)  # Comment out if labels are already one-hot encoded

# Define the model architecture (simple example)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(42,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))  # Adjust for number of unique labels

# Compile the model (adjust optimizer and loss as needed)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32)

# Save the trained model (optional)
model.save('sign_language_model.h5')

print("Model trained and saved!")