# 🐾 End-to-End CNN Image Classification Project — Cat vs Dog Prediction

A complete Deep Learning project using Convolutional Neural Networks (CNNs) to classify cat and dog images with high accuracy and deploy it using Streamlit.

# 🧭 Project Overview

This project focuses on building and deploying a Convolutional Neural Network (CNN) model capable of distinguishing between cats and dogs from image inputs.

It demonstrates a full deep learning pipeline — from data preprocessing and model training to evaluation and deployment using Streamlit, allowing users to upload an image and instantly see the prediction.

## ⚙️ Tech Stack

Category	                             Tools & Libraries

Language	                                 Python 3.x
Deep Learning Framework	             TensorFlow / Keras
Libraries	                          NumPy, Pandas, Matplotlib
Deployment	                                 Streamlit
IDE	                                 VS Code, Jupyter Notebook
Dataset	                         keras Cats and Dogs Dataset (25,000 images)


## 🔄 End-to-End Process Flow

1️⃣ Data Loading & Preprocessing

Imported dataset containing labeled images of cats and dogs.

Resized all images to a consistent shape (e.g., 64×64×3).

Normalized pixel values to scale between 0 and 1.

Used data augmentation (ImageDataGenerator) for better generalization.

2️⃣ Model Architecture (CNN)

Designed a Convolutional Neural Network with:

Conv2D layers (feature extraction)

MaxPooling layers (dimensionality reduction)

Flatten layer

Dense layers (classification)

Final layer: 1 neuron (Sigmoid) for binary classification.

3️⃣ Model Compilation & Training

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

Trained model on ~20,000 images for multiple epochs.

Achieved accuracy > 95% on validation set.

4️⃣ Model Evaluation

Plotted accuracy/loss curves to monitor overfitting.

Evaluated performance on test set with confusion matrix and classification metrics.

5️⃣ Model Saving

Saved trained model for deployment:

model.save("cat_dog_model.h5")

6️⃣ Streamlit Deployment

Built a simple and elegant Streamlit web app for real-time image prediction:

Users upload an image (.jpg or .png)

App preprocesses and feeds image to CNN model

Displays prediction: Cat 🐱 or Dog 🐶 with confidence score

# 📈 Results

Metric	            Value

Training         Accuracy	97%
Validation       Accuracy	95%
Test Accuracy	     94%
Model File	     cat_dog_model.h5

# Sample Output:

Uploaded image → “Dog 🐶 (Confidence: 97%)”

Uploaded image → “Cat 🐱 (Confidence: 96%)”



# 💻 Streamlit App

Run the Streamlit app locally:

streamlit run app.py


# Features:

Upload any image (.jpg/.png)

Automatic preprocessing (resize, normalize)

Instant prediction with high accuracy

Clean, responsive, and user-friendly UI

# 🧠 Key Learnings

Implemented CNN for image classification

Applied data augmentation and normalization

Tuned hyperparameters for improved model performance

Deployed a Streamlit app for interactive prediction

Learned end-to-end deep learning workflow — from model design to user-ready deployment and moduler coding

# 🚀 Future Improvements

Add Grad-CAM visualization to highlight model attention areas

Deploy on Streamlit Cloud / Hugging Face Spaces

Include multi-class prediction (e.g., more animals)

Integrate confidence thresholding for uncertain predictions

# 👨‍💻 Author

👤 [Onkar Patil]

Data Scientist | Deep Learning Enthusiast

📧 [opatil345@gmail.com]

🔗 LinkedIn: www.linkedin.com/in/opatil345/
