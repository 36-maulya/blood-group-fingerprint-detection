Blood Group Detection from Fingerprint Images
Project Overview

This project is a deep learning-based system that predicts a person’s blood group from their fingerprint images.
The system uses a Convolutional Neural Network (CNN) to learn fingerprint patterns associated with blood groups A, B, AB, and O.

The project is implemented entirely in Google Colab and includes a Flask app that allows users to upload fingerprint images and get live predictions.

Motivation

Traditional blood group determination requires blood samples.

Using fingerprints for prediction provides a non-invasive alternative.

Helps in learning deep learning, computer vision, and deployment techniques.

Dataset

Fingerprint images labeled with blood groups.

Preprocessing includes:

Resizing images to 128x128 pixels.

Normalizing pixel values to the range [0,1].

Categorical encoding for blood group classes.

Training/validation split: 80:20

Folder structure for training:

dataset/
    A/
    B/
    AB/
    O/

Technologies Used

Python – Programming language

TensorFlow / Keras – For CNN model building and training

Flask – Web app for uploading images and displaying predictions

ngrok – To expose Flask app with a public URL

Google Colab – Cloud environment for GPU training

Methodology

Data Preprocessing:

Resize images to 128x128 pixels.

Normalize pixel values.

Convert blood group labels to categorical.

CNN Architecture:

3 Convolutional layers with ReLU activation.

MaxPooling layers to reduce dimensions.

Flatten layer followed by Dense layers for classification.

Dropout layer to prevent overfitting.

Softmax activation for multi-class output.

Model Training:

Loss: categorical_crossentropy

Optimizer: Adam

Epochs: 20 (adjustable)

Batch size: 32

Validation split: 20%

Model Saving & Loading:

Saved as bloodgroup_cnn.h5

Loaded in Flask app for predictions

Flask Web App:

Users upload fingerprint images.

App predicts blood group using CNN.

Live URL via ngrok.

Results

Validation Accuracy: [Insert your accuracy, e.g., 92%]

Example predictions:

Fingerprint Image	Predicted Blood Group
fingerprint1.jpg	A
fingerprint2.jpg	AB

Accuracy may vary depending on the dataset size and fingerprint quality.

How to Run

Clone the repository:

git clone <your_repo_link>


Open the Colab notebook and run all cells:

Upload dataset

Preprocess images

Train CNN model

Launch Flask app via ngrok

Access the live URL from ngrok to upload fingerprints and see predictions.

Future Improvements

Increase dataset size to improve accuracy.

Use data augmentation to enhance model generalization.

Experiment with pretrained models (e.g., MobileNet, ResNet).

Improve Flask frontend for better user experience.
