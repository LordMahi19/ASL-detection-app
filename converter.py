import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model('sign_language_model.h5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('sign_language_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite!")