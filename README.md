# Hand Gesture Recognition Project

## Overview
This repository contains a machine learning project focused on recognizing hand gestures using a deep learning model. The project utilizes a dataset of 20 different gesture classes, sourced from the Kaggle "Hand Gesture Recognition Dataset," to train a Convolutional Neural Network (CNN) for classifying gestures from images. The model is designed to predict gestures accurately, with the output mapped to meaningful gesture names rather than numerical indices, enhancing usability. The project includes both training and testing phases, with the ability to save and load model weights for future use.

## Libraries
- **PyTorch**: For building and training the CNN model.
- **Torchvision**: For image preprocessing and transformations.
- **NumPy**: For numerical operations and array handling.
- **OpenCV (cv2)**: For image loading and processing.
- **Scikit-learn**: For label encoding and evaluation metrics.
- **Tqdm**: For progress bar visualization during training.
- **Pathlib**: For robust file path handling.

## Project Logic Highlights
- **Data Preparation**: Loads and preprocesses a dataset of 24,000 images split into 20 gesture categories, ensuring proper resizing and normalization.
- **Model Architecture**: Implements a custom CNN with multiple convolutional and pooling layers, followed by fully connected layers for classification.
- **Training Process**: Trains the model using a defined number of epochs with early stopping and learning rate scheduling to optimize performance.
- **Model Saving**: Saves the best model weights based on validation accuracy and a final model with metadata for testing.
- **Testing and Prediction**: Loads the trained model to predict gestures on new images, mapping numerical predictions to gesture names using a custom dictionary.
- **Error Handling**: Includes checks for file existence and image loading to ensure robust execution.
- **Gesture Mapping**: Converts predicted indices to human-readable gesture names, customizable based on the dataset's folder contents.

This project provides a foundation for gesture recognition applications, with flexibility to adapt the gesture mapping to specific use cases or datasets.
