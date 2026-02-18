**Blood Group Detection from Fingerprint Images**

**Project Overview**

This project is a deep learning-based system that predicts a person’s blood group from their fingerprint images using a Convolutional Neural Network (CNN).

Users can upload fingerprint images via a web interface, and the model predicts one of the four blood groups: A, B, AB, or O.

Backend: CNN model in TensorFlow/Keras

Frontend: Flask web app

Deployment: Google Colab + ngrok (or future serverless deployment)

**Motivation**

Traditional blood group detection requires blood samples.

Using fingerprints provides a non-invasive alternative.

Learn deep learning, computer vision, and web deployment in one project.

**Dataset**

Fingerprint images labeled with blood groups.

Preprocessing includes:

Resizing to 128x128 pixels

Normalizing pixel values to [0,1]

One-hot encoding blood group labels

**Folder structure:**

dataset/
    A/
    B/
    AB/
    O/

**Technologies Used**

Python – Programming language

TensorFlow / Keras – CNN model development

Flask – Web application for predictions

ngrok – Expose local Flask app as public URL

Google Colab – GPU-enabled cloud environment

**Methodology**
1️⃣ **Data Preprocessing**

Resize all images to 128x128 pixels.

Normalize pixel values.

Split dataset into training (80%) and validation (20%).

2️⃣ **CNN Architecture**

3 Convolutional layers with ReLU activation

MaxPooling layers for dimensionality reduction

Flatten layer → Dense layers → Dropout (0.5)

Softmax activation for multi-class classification

Model Summary Snapshot:

Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 126, 126, 32)    896
max_pooling2d (MaxPooling2D) (None, 63, 63, 32)      0
...
dense_2 (Dense)               (None, 4)               516

3️⃣ **Model Training**

Loss: categorical_crossentropy

Optimizer: Adam

Epochs: 20 (adjustable)

Batch size: 32

Training Graphs (Example Placeholder):


4️⃣ **Model Evaluation**

Validation Accuracy: [Insert your accuracy, e.g., 92%]

Predictions Example:

Fingerprint	Predicted Blood Group
fingerprint1.jpg	A
fingerprint2.jpg	AB

**Web Application**

Upload fingerprint images via Flask app.

Model predicts blood group in real-time.

ngrok URL provides public access (temporary live link).

App Screenshot (Placeholder):

⚡ How to Run

Clone the repository:

git clone <your_repo_link>


Open the Colab notebook and run all cells:

Upload dataset

Preprocess images

Train CNN model

Launch Flask app via ngrok

Access the live URL to upload fingerprint images and see predictions.

🔮 Future Improvements

Add data augmentation to improve generalization.

Experiment with pretrained CNN models (e.g., MobileNet, ResNet).

Optimize Flask frontend for better user experience.

Deploy backend as serverless API for Vercel or other cloud hosting.
