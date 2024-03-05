# Jude Tear
# Feb 7th 2024

import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

# Normalize all the data 
def normalizeData(data, multiple):
    output = []

    if multiple:
        print("\nNormalizing data within multiple tasks\n")

        for i in range(len(data)):
            X_train = data[i][0]
            X_test = data[i][2]

            # Normalize images
            normalized_X_train = ((X_train - X_train.min()) / 
                        (X_train.max() - X_train.min()) * 255).astype(np.uint8)
            normalized_X_test = ((X_test - X_test.min()) / 
                        (X_test.max() - X_test.min()) * 255).astype(np.uint8)

            output.append([normalized_X_train, data[i][1], normalized_X_test, data[i][3]])

    else:
        print("\nNormalizing one task\n")

        X_train = data[0]
        X_test = data[2]

        # Normalize images
        normalized_X_train = ((X_train - X_train.min()) / 
                        (X_train.max() - X_train.min()) * 255).astype(np.uint8)
        normalized_X_test = ((X_test - X_test.min()) / 
                        (X_test.max() - X_test.min()) * 255).astype(np.uint8)

        output = [normalized_X_train, data[1], normalized_X_test, data[3]]

    return output

# Resize images
def resize_images(data, width, height):
    resized_data = np.empty((data.shape[0], height, width, data.shape[3]))
    for i, img in enumerate(data):
        resized_data[i] = cv.resize(img, (width, height))
    return resized_data

# Scale image pixel values
def scale_data(X_train, X_test):
    
    # Scale so values fall between 0 and 1
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    return X_train_scaled, X_test_scaled

# Filter data randomly NOT preserving class ratio
def filter_data_rand(X, Y, train_percent):
    """
    Splits the dataset into a training set and a remaining set based on the specified percentage.
    
    Parameters:
    - X: numpy array, the complete dataset.
    - Y: numpy array, the labels corresponding to the dataset.
    - train_percent: float, the percentage of the dataset to be used for training (between 0 and 100).
    
    Returns:
    - X_train: numpy array, the randomly selected training data.
    - y_train: numpy array, the labels corresponding to the randomly selected training data.
    """
    # Calculate the number of samples to be used for training
    num_train_samples = int(len(X) * (train_percent / 100.0))
    
    # Generate a random permutation of indices from 0 to the length of the dataset
    shuffled_indices = np.random.permutation(len(X))
    
    # Select indices for training and the remaining data
    train_indices = shuffled_indices[:num_train_samples]
    
    # Split the dataset for X
    X_train = X[train_indices]
    
    # Split the dataset for Y using the same indices
    y_train = Y[train_indices]
    
    return X_train, y_train

# Filter data and PRESERVE class ratio
def filter_data_strat(X, Y, percent):
    """
    Splits dataset into a training set and a remaining set based on the specified percentage

    Parameters:
    - X: numpy array, the complete train data set
    - Y: 
    """
    percent = percent/100
    X_train, X_remainder, Y_train, Y_remainder = train_test_split(X, Y, train_size=percent, stratify=Y, random_state=42)
    return  X_train, Y_train

# Make data binary
def makeDataBinary(y_train, y_test):

    # Convert the labels for the train set
    y_train_binary = np.array([ [y[1], max(y[0], y[2])] for y in y_train ])
    
    # Convert the labels for the test set
    y_test_binary = np.array([ [y[1], max(y[0], y[2])] for y in y_test ])
    
    return y_train_binary, y_test_binary