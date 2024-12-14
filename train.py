# File 2: train_model.py
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf

if __name__ == '__main__':
    # Load the dataset
    dataset = np.load("sign_language_dataset.npz")
    data = dataset["data"]
    labels = dataset["labels"]

    # Convert to TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    tf_dataset = tf_dataset.shuffle(len(data)).batch(32)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(data))
    train_dataset = tf_dataset.take(train_size)
    test_dataset = tf_dataset.skip(train_size)

    # Create and train a TensorFlow Decision Forests model
    model = tfdf.keras.RandomForestModel()
    model.fit(x=train_dataset)

    # Evaluate the model
    evaluation = model.evaluate(test_dataset)
    print(f"Evaluation results: {evaluation}")

    # Save the TensorFlow model
    model.save("sign_language_model")

    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_saved_model("sign_language_model")
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open("sign_language_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("TensorFlow Lite model saved as sign_language_model.tflite")
