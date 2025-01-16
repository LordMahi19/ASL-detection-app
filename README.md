# Sign Language Detection with TensorFlow

This project is a continuation of this [project](https://github.com/LordMahi19/ASL-detection) where we have this [Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/data) to train a machine learning that predicts sign language charachters using mediapipes handlarmark library. The trained model was then converted to TensorFlow Lite format for efficient deployment on mobile devices. We will make an android app that will use this tensorflow lite to predict the sign language charachters. For a comprehensive understanding, please visit the previous repository: [previous repo](https://github.com/LordMahi19/ASL-detection).

## Changes and Improvements

These are the key changes compared to the previos repository:

### Key Changes:

- **Dataset Format**: Transitioned from a pickle dataset to an NPZ dataset for better compatibility and performance.
- **Model Format**: Developed the model in `.h5` format instead of `.p` for improved functionality.
- **Model Conversion**: Converted the `.h5` model to `.tflite` format, making it suitable for mobile deployment. We have used this model in this [project](https://github.com/LordMahi19/ASL-detection-android)
- **Labels File**: Created a text file containing all the labels, with each label on a new line, maintaining the same order as during training. This will also be used the android app.

## Sign Language Characters

Below is a visual representation of the sign language characters recognized by our model:

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

3. **run the model
   ***run the following script in your terminal. 
   ```bash
   python test.py
   ```

## Contributing

We welcome contributions to enhance the functionality and performance of this app. Feel free to fork the repository, make your changes, and submit a pull request.
