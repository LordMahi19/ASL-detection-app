# Android App for Sign Language Detection with TensorFlow Lite

Welcome to the **Sign Language Detection App** project! This application leverages the power of **TensorFlow** to recognize and interpret sign language characters.

## Overview

In this project, we utilized the dataset from this [repository](https://github.com/LordMahi19/ASL-detection) to train a machine learning model capable of predicting sign language characters. The trained model was then converted to TensorFlow Lite format for efficient deployment on mobile devices. For a comprehensive understanding, please visit the [original repository](https://github.com/LordMahi19/ASL-detection).

## Changes and Improvements

To enhance the performance and usability of our application, we implemented several key changes and improvements:

### Key Changes:

- **Dataset Format**: Transitioned from a pickle dataset to an NPZ dataset for better compatibility and performance.
- **Model Format**: Developed the model in `.h5` format instead of `.p` for improved functionality.
- **Model Conversion**: Converted the `.h5` model to `.tflite` format, making it suitable for mobile deployment.
- **Labels File**: Created a text file containing all the labels, with each label on a new line, maintaining the same order as during training.

## Sign Language Characters

Below is a visual representation of the sign language characters recognized by our app:

![Sign Language Characters](./signs.png)

## Getting Started

To get started with the Sign Language Detection App, follow these steps:

1. **Clone the Repository**: Clone the project repository to your local machine.

   ```bash
   git clone https://github.com/LordMahi19/ASL-detection-app.git
   ```

2. **Install Dependencies**: Ensure you have all the necessary dependencies installed.
   ```bash
   pip install -r requirements.txt
   ```

## Contributing

We welcome contributions to enhance the functionality and performance of this app. Feel free to fork the repository, make your changes, and submit a pull request.
