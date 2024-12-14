import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = np.load("sign_language_dataset.npz")
X_train = data['data']
y_train = data['labels']

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], 42)

# Label Encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# One-hot encode the integer labels
num_classes = len(le.classes_)  # Get the number of unique classes
y_train_encoded = to_categorical(y_train_encoded, num_classes=num_classes)

# Define the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(42,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) # Use num_classes here

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32)

# Save the model
model.save('sign_language_model.h5')

print("Model trained and saved!")