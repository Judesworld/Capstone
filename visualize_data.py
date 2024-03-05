# This file is used for visualizing the data

import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import numpy as np

def generate_image_grid(data, start_idx, num_images, rows, cols, image_height, image_width):
    grid_image = np.zeros((rows * image_height, cols * image_width, 3), dtype=np.uint8)
    for i in range(num_images):
        idx = start_idx + i
        if idx < len(data):
            row = i // cols
            col = i % cols
            grid_image[row * image_height:(row + 1) * image_height,
                       col * image_width:(col + 1) * image_width] = data[idx]
    return grid_image

def display_images(subsets, num_images=25, rows=5, cols=5):
    for subset_name, subset_array in subsets.items():
        if len(subset_array) < num_images:
            print(f"Not enough images in {subset_name} to display {num_images} images.")
            continue

        image_height, image_width = subset_array[0].shape[:2]
        grid_image = generate_image_grid(subset_array, 0, num_images, rows, cols, image_height, image_width)
        cv.imshow(f'First 25 Images of {subset_name}', grid_image)
        
        # Wait for 'q' to close the window
        while True:
            key = cv.waitKey(0)
            if key in [ord('q'), ord('Q')]:
                break
        cv.destroyAllWindows()

def visualize_results(training_sizes, training_times, test_accuracies, test_losses, sensitivity_scores, specificity_scores, f1_scores):
    """
    Visualize the results of the machine learning experiment.
    
    Parameters:
    - training_sizes: List of training sizes.
    - training_times: Dictionary of training times for each training size.
    - test_accuracies: Dictionary of test accuracies for each training size.
    - test_losses: Dictionary of test losses for each training size.
    - sensitivity_scores: Dictionary of sensitivity scores for each training size.
    - specificity_scores: Dictionary of specificity scores for each training size.
    - f1_scores: Dictionary of F1 scores for each training size.
    """
    plt.figure(figsize=(10, 7))

    # Plot each metric in a subplot
    metrics = [training_times, test_accuracies, test_losses, sensitivity_scores, specificity_scores, f1_scores]
    metric_labels = ['Training Times', 'Test Accuracies', 'Test Losses', 'Sensitivity Scores', 'Specificity Scores', 'F1 Scores']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels), 1):
        plt.subplot(2, 3, i)
        plt.plot(training_sizes, [metric[size] for size in training_sizes], label=label)
        plt.xlabel('Training Size')
        plt.ylabel(label)
        plt.title(label)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using seaborn's heatmap.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_incorrect_images_bin(X_test, incorrect_indices, y_pred, y_test_binary):
    # Loop through the incorrect indices
    for index in incorrect_indices:
        plt.figure(figsize=(10, 10))  # Adjust the size as needed
        plt.imshow(X_test[index])  # Assuming X_test is preprocessed image data
        plt.title(f"Predicted: {'Damaged' if y_pred[index] == 1 else 'Intact'}, Actual: {'Damaged' if y_test_binary[index] == 1 else 'Intact'}")
        plt.show()

def visualize_incorrect_images_multiclass(X_test, incorrect_indices, y_pred_classes, y_true_classes):
    class_names = ['Intact', 'Partial', 'Global']  # Class 0, 1, 2 respectively
    
    for index in incorrect_indices:
        plt.figure(figsize=(10, 10))  # Adjust size as needed
        plt.imshow(X_test[index])  # Assuming X_test is preprocessed image data
        predicted_label = class_names[y_pred_classes[index]]
        true_label = class_names[y_true_classes[index]]
        plt.title(f"Predicted: {predicted_label}, Actual: {true_label}")
        plt.show()

def visualize_images_with_labels(X, y, indices):
    """
    Visualize images with their associated labels.

    Parameters:
    - X: The dataset containing the images.
    - y: The labels associated with the images.
    - indices: The indices of the images to be visualized.
    """
    # Define the label mapping
    label_mapping = {
        (1, 0, 0): 'global',
        (0, 1, 0): 'intact',
        (0, 0, 1): 'partial',
    }
    
    plt.figure(figsize=(15, 10))  # Set the figure size for better visibility
    columns = 5  # Adjust the number of columns based on your preference
    for i, idx in enumerate(indices):
        plt.subplot(len(indices) // columns + 1, columns, i + 1)  # Create a subplot for each image
        plt.imshow(X[idx])  # Display the image
        # Convert the label to a tuple to use as a key for the label mapping
        label_tuple = tuple(y[idx])
        label_text = label_mapping.get(label_tuple, "Unknown")  # Get the corresponding text label
        plt.title(f'Label: {label_text}')  # Set the title as the text label
        plt.axis('off')  # Turn off the axis for a cleaner look
    plt.tight_layout()
    plt.show()



def visualize_images_with_labels_bin(X, y, indices):
    """
    Visualize images with their associated binary labels.

    Parameters:
    - X: The dataset containing the images.
    - y: The labels associated with the images, assumed to be binary in this context.
    - indices: The indices of the images to be visualized.
    """
    # Adjust the label mapping for binary classification
    label_mapping = {
        (1, 0): 'intact',  # Binary label for intact
        (0, 1): 'damaged',  # Binary label for damaged (combines global and partial from previous ternary setup)
    }
    
    plt.figure(figsize=(15, 10))  # Set the figure size for better visibility
    columns = 5  # Adjust the number of columns based on your preference
    for i, idx in enumerate(indices):
        plt.subplot(len(indices) // columns + 1, columns, i + 1)  # Create a subplot for each image
        plt.imshow(X[idx])  # Display the image. Assume X[idx] is correctly formatted (e.g., RGB or grayscale)
        
        # Ensure the label is a tuple for the mapping (this assumes y[idx] is in the correct binary format)
        if isinstance(y[idx], list):  # If the labels are lists, convert to tuples
            label_tuple = tuple(y[idx])
        elif isinstance(y[idx], (np.ndarray, list)):  # If labels are numpy arrays or lists, convert to tuples
            label_tuple = tuple(y[idx].tolist())
        else:
            label_tuple = (y[idx],)  # For single binary labels, ensure they are in tuple form
        
        label_text = label_mapping.get(label_tuple, "Unknown")  # Get the corresponding text label
        plt.title(f'Label: {label_text}')  # Set the title as the text label
        plt.axis('off')  # Turn off the axis for a cleaner look
    plt.tight_layout()
    plt.show()


def show_image_with_label(X, y, index):
    """
    Show a single image and its associated label.

    Parameters:
    - X: The dataset containing the images.
    - y: The labels associated with the images.
    - index: The index of the image and label to be visualized.
    """
    # Display the image
    plt.imshow(X[index])
    plt.title(f'Label: {y[index]}')
    plt.axis('off')  # Hide the axes for better visualization
    plt.show()