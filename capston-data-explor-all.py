import cv2 as cv 
import numpy as np

# Returns list of lists with either one or multiple lists
def getData(num, multiple):

    if multiple:
        print("\nMultiple tasks selected")

        output = []
        classes = []
        for i in range(len(num)):
            X_train_file = f"task{num[i]}/task{num[i]}"+"_X_train.npy"
            y_train_file = f"task{num[i]}/task{num[i]}"+"_y_train.npy"

            X_test_file = f"task{num[i]}/task{num[i]}"+"_X_test.npy"
            y_test_file = f"task{num[i]}/task{num[i]}"+"_y_test.npy"

            X_train = np.load(X_train_file)
            y_train = np.load(y_train_file)
            X_test  = np.load(X_test_file)
            y_test = np.load(y_test_file)

            # Get the number of classes within the task
            num_classes = len(y_train[0])

            # Contain data within list
            container = [X_train, y_train, X_test, y_test]

            output.append(container)
            classes.append(num_classes)

        return output, classes

    else:
        print("Single task selected")
        X_train_file = f"task{num[0]}/task{num[0]}"+"_X_train.npy"
        y_train_file = f"task{num[0]}/task{num[0]}"+"_y_train.npy"

        X_test_file = f"task{num[0]}/task{num[0]}"+"_X_test.npy"
        y_test_file = f"task{num[0]}/task{num[0]}"+"_y_test.npy"

        X_train = np.load(X_train_file)
        y_train = np.load(y_train_file)
        X_test  = np.load(X_test_file)
        y_test = np.load(y_test_file)

        # Get the number of classes
        num_classes = len(y_train[0])
        classes = [num_classes]

        return [X_train, y_train, X_test, y_test], classes

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


# Helper Function - Generate subsets
def create_subsets(classes):
    subsets = {}

    for num_classes in classes:
        for i in range(num_classes):
            # Create a binary string representing the class
            binary_str = '0' * num_classes
            subset_name = 'subset_' + binary_str[:i] + '1' + binary_str[i+1:]
            
            # Create the subset list using exec
            exec(f"{subset_name} = []")
            
            # Store the reference to the list in the subsets dictionary
            subsets[subset_name] = eval(subset_name)

    return subsets

def divide(data, multiple, classes):
    def label_to_binary_str(label):
        return ''.join(str(int(bit)) for bit in label)

    all_subsets = []

    if multiple:
        print("Multi-task - Organizing by classes\n")

        for task_index in range(len(data)):
            X_train = data[task_index][0]
            y_train = data[task_index][1]

            # Create subsets for this task
            task_subsets = create_subsets([classes[task_index]])

            # Divide data into subsets
            for i, label in enumerate(y_train):
                label_str = label_to_binary_str(label)
                subset_name = 'subset_' + label_str
                if subset_name in task_subsets:
                    task_subsets[subset_name].append(X_train[i])

            # Convert lists to numpy arrays
            for key in task_subsets:
                task_subsets[key] = np.array(task_subsets[key])

            all_subsets.append(task_subsets)

    else:
        print("Single Task - Organizing by classes\n")

        X_train = data[0]
        y_train = data[1]

        # Create subsets for the number of classes in the single task
        task_subsets = create_subsets([classes[0]])

        # Divide data into subsets
        for i, label in enumerate(y_train):
            label_str = label_to_binary_str(label)
            subset_name = 'subset_' + label_str
            if subset_name in task_subsets:
                task_subsets[subset_name].append(X_train[i])

        # Convert lists to numpy arrays
        for key in task_subsets:
            task_subsets[key] = np.array(task_subsets[key])

        all_subsets.append(task_subsets)

    return all_subsets

# Create a function to generate the grid image
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

def display_images_in_grid(subsets, num_images, rows, cols):
    for subset_name, images in subsets.items():
        if len(images) == 0:
            continue  # Skip if no images in the subset
        
        # Assuming all images in the subset are of the same size
        image_height, image_width = images[0].shape[:2]

        # Generate grid image
        grid_image = generate_image_grid(images, 0, num_images, rows, cols, image_height, image_width)

        # Display the grid
        cv.imshow(f'Subset: {subset_name}', grid_image)
        
        # Wait for the 'q' key to be pressed to close the window
        while True:
            key = cv.waitKey(0)
            if key == ord('q') or key == ord('Q'):
                break

        cv.destroyAllWindows()

def visualize_results(result, num_images, rows, cols):
    if isinstance(result, list):  # Multiple tasks scenario
        for task_subsets in result:
            display_images_in_grid(task_subsets, num_images, rows, cols)
    else:  # Single task scenario
        display_images_in_grid(result, num_images, rows, cols)

def resize_images(data, width, height):
    resized_data = np.empty((data.shape[0], height, width, data.shape[3]))
    for i, img in enumerate(data):
        resized_data[i] = cv.resize(img, (width, height))
    return resized_data


if __name__ == "__main__":
    # Outline 
    # Pre-req: desired tasks, num_images and rows+cols

    # 1. Use file path to get train/test data and labels (Hardcode)
    # 2. Normalize and convert to uint8 from float 32
    # 3. Seperate data into subgroups based label associated with data
    # 4. Display the first n images in a grid for each class within the task

    # Status:
    # - Done
    
    multiple = bool

    # ***INPUT DESIRED TASK #'S HERE***
    tasks = [5]

    num_images = 12
    cols = 4
    rows = 3

    if len(tasks) == 1:
        multiple = False
    else:
        multiple = True

    # Expecting back either one or multiple batches of task data
    data, classes = getData(tasks, multiple)

    # Expect data to be normalized and converted to correct format
    data = normalizeData(data, multiple)

    # Assuming all images are of the same size  
    # image_height, image_width = data[0][0].shape[1:3] ## HARDCODE
    # print(image_height, image_width)

    results = divide(data, multiple, classes)
    # visualize_results(results, num_images, rows, cols)

    # TASK 5
    from models_utils import train_inception_v3
    from models_utils import train_resnet50
    from models_utils import train_efficientNet
    from models_utils import train_mobileNet
    import time

    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]

    #### Run the InceptionV3 ####
    # Time: 670.69 == 11.18 minutes
    # Test Loss: 1.349
    # Test Accuracy: 0.534

    # start = time.time()
    # train_inception_v3(X_train, y_train, X_test, y_test, True)
    # end = time.time()
    # print(end - start)


    #### Run the ResNet-50 ####
    # Time: 770.38s == 12.84 minutes
    # Test Loss: 2.176
    # Test Accuracy: 0.685

    # start = time.time()
    # train_resnet50(X_train, y_train, X_test, y_test, False)
    # end = time.time()
    # print(end - start)


    #### Run the EfficientNetB0 ####
    # Time: 135.13 == 2.25 minutes
    # Test Loss: 0.550
    # Test Accuracy: 0.747

    start = time.time()
    train_efficientNet(X_train, y_train, X_test, y_test, False)
    end = time.time()
    print(end - start)
    

    #### Run the MobileNet ####
    # Time: 166.69 == 2.78 minutes
    # Test Loss: 3.840
    # Test Accuracy: 0.644

    # start = time.time()
    # train_mobileNet(X_train, y_train, X_test, y_test, False)
    # end = time.time()
    # print(end - start)
    


    




