# Android app for sign language detection with tensorflow lite

In this project we have used the same dataset that we use earlier in this [reposiotory](https://github.com/LordMahi19/ASL-detection) to train a model a model to predict sign language charachters and then we made an android app that inferences that model locally using tensorflow light. please see the this [reposiotory](https://github.com/LordMahi19/ASL-detection) first.

We have introduce fundamental changes to achieve our goal. We are using the same technique like before; collecting hand landmarks x and y coordinates and and label of each image to make a dataset and then training a prediction model using that dataset using randorm forrest classifier.

Instead of creating a pickle dataset we have created an npz dataset. And then we have created an .h5 model instead of .p

We then convert that to .tflite model to use in our android app. We have also created a text file with all the labels written inside of it one label per line in the exact same order as they were during the training.

**App making**

- We have copied the model.tflite file and the labels.txt file inside the asset folder within our android directory.
- The HandLandmarkerHelper.kt file processes the images and uses mediapipe to extract hand landmark information.
- TFLiteHelper.kt loads and sets everyting to inference the model.tflite
- OverLayview.kt draws the landmark connections, makes predictions using the TFLiteHelper.kt class and displays the predicted class.

[**Dowload the app**](https://lut-my.sharepoint.com/:u:/g/personal/mahi_talukder_student_lut_fi/EQjG3WvGn7hGttbHXBUil6IB_NdItEkd-Z19qzHSJObz1A?e=jhvUki)
