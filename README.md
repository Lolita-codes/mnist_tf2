# MNIST Classification

## Project Description
This project aims to build and train a machine learning model to classify handwritten digits from the MNIST dataset. The MNIST dataset is a large collection of grayscale images of handwritten digits (0-9), commonly used as a benchmark for image classification tasks.

## Project Objective
The primary objective is to develop a model that can accurately predict the digit represented in an unseen handwritten digit image. This involves training a model on a subset of the MNIST dataset and evaluating its performance on a separate test set.

## Project Overview
This project follows a typical machine learning workflow:
- Data Acquisition: The MNIST dataset is loaded using TensorFlow Datasets.
- Data Preprocessing: The images are scaled to a range of 0-1 and the dataset is split into training, validation, and test sets.
- Model Building: A sequential neural network model is constructed using Keras, consisting of an input layer, two hidden layers with ReLU activation, and an output layer with softmax activation.
- Model Training: The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function.
- Model Evaluation: The trained model's performance is assessed on the test set using metrics such as loss and accuracy.
  
## Methods
- Supervised Learning
- Neural Networks
- Backpropagation

## Technologies Used
- Python  
- TensorFlow
- Keras
- NumPy

# Key Findings
- The trained model achieves a high accuracy of 96% on the MNIST test set.
- The model demonstrates the ability to generalize well to unseen data, indicating its effectiveness in classifying handwritten digits.
- Deep learning techniques, particularly neural networks, are effective for image classification tasks like handwritten digit recognition.
