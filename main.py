# Jude Tear
# Data-Loading File

import cv2 as cv
import numpy as np
import time

def load_and_normalize_data():
    # Load data
    train_data = np.load('task5/task5_X_train.npy')
    train_label = np.load('task5/task5_y_train.npy')
    test_data = np.load('task5/task5_X_test.npy')
    test_label = np.load('task5/task5_y_test.npy')

    # Find global min and max from training data for normalization
    global_min = train_data.min()
    global_max = train_data.max()

    # Normalize train and test data
    normalized_train_data = ((train_data - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    normalized_test_data = ((test_data - global_min) / (global_max - global_min) * 255).astype(np.uint8)

    return normalized_train_data, train_label, normalized_test_data, test_label

def segregate_data(normalized_train_data, train_label):
    subsets = {'subset_001': [], 'subset_010': [], 'subset_100': []}
    for i, label in enumerate(train_label):
        label_str = ''.join(str(int(bit)) for bit in label)
        subset_key = f'subset_{label_str}'
        subsets[subset_key].append(normalized_train_data[i])
    
    # Convert lists to numpy arrays
    for key in subsets:
        subsets[key] = np.array(subsets[key])
    return subsets



if __name__ == "__main__":
    from mobileNet_model import MobileNetTrainer

    from visualize_data import display_images, visualize_incorrect_images_bin
    from visualize_data import  visualize_incorrect_images_multiclass
    from visualize_data import visualize_images_with_labels
    from visualize_data import visualize_images_with_labels_bin
    from visualize_data import show_image_with_label

    from data_filtration import makeDataBinary


    X_train, y_train, X_test, y_test = load_and_normalize_data()
    subsets = segregate_data(X_train, y_train)

    # Display the counts of images in each subset
    # for subset_name, subset_array in subsets.items():
    #     print(f"{subset_name} contains {len(subset_array)} images.")

    # Display images from the subsets --
    # display_images(subsets) 

    ############ Testing, Running and Evaluating Models ############
    # 
    # MobileNet (Multi-classification and Binary)

    # Multiclass *****
    # multiclass_trainer = MobileNetTrainer(num_classes=3, model_type='multiclass', save_model=False, width=224, height=224)
    # incorrect_idx, y_pred, y_true = multiclass_trainer.train(X_train, y_train, X_test, y_test, resize=False)
    # visualize_incorrect_images_multiclass(X_test, incorrect_idx, y_pred, y_true)

    # Binary (merge partial and global classes) *****
    y_train_binary, y_test_binary = makeDataBinary(y_train, y_test)

    # binary_trainer = MobileNetTrainer(num_classes=2, width=224, height=224, model_type='binary', save_model=False)
    # incorrect_idx, y_pred, y_true = binary_trainer.train(X_train, y_train_binary, X_test, y_test_binary, resize=False) 
    # visualize_incorrect_images_bin(X_test, incorrect_idx, y_pred, y_true)
    # print(incorrect_idx)


    incorrect_idx = [70, 71, 75, 81, 83, 85, 86, 89, 90, 91, 97, 99, 100, 101, 105, 107, 119, 121, 123, 129, 135, 136, 138, 139, 141, 142, 143]
    # visualize_images_with_labels_bin(X_test, y_test_binary, incorrect_idx)
    show_image_with_label(X_test, y_test, 71)
    cv.imshow('Image', X_test[71])
    cv.waitKey(10000)
    cv.destroyAllWindows()