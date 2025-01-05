import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Prepare input
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

sample_input = np.random.rand(1, 42).astype(np.float32)  # Replace with actual data
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output)
print(f"Predicted class: {predicted_class}")
