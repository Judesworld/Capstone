import cv2 as cv 
import numpy as np
import tensorflow as tf
from data_filtration import normalizeData

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

# Helper Functions - Generate subsets
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

# Divide the data up
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

# Generate a grid for the images to display
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

# Visualize images
def visualize_results(result, num_images, rows, cols):
    if isinstance(result, list):  # Multiple tasks scenario
        for task_subsets in result:
            display_images_in_grid(task_subsets, num_images, rows, cols)
    else:  # Single task scenario
        display_images_in_grid(result, num_images, rows, cols)

# Test code to run the experiment
def run_experiment(X_train, y_train, X_test, y_test, train_sizes, split_func, model):
    """
    Runs model training experiments over different train sizes using specified split function.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Test data and labels.
    - train_sizes: List of integers representing the percentages of data to use.
    - split_func: Function to use for splitting the data (e.g., filter_data_rand or filter_data_strat).

    Returns:
    - A dictionary containing training times keyed by the train size percentage.
    """
    training_times = {}
    test_accuracies = {}
    test_losses = {}
    specificity_scores = {}
    sensitivity_scores = {}
    f1_scores = {}


    for size in train_sizes:
        # Split the dataset according to the specified size
        X_train_subset, y_train_subset = split_func(X_train, y_train, size)

        # Record the start time of training
        start_time = time.time()

        # Train the model using the subset of training data
        # Adjust this line to match the signature of your actual training function
        test_acc, test_loss, specificity, sensitivity, f1 = model(X_train_subset,
                                                                y_train_subset, 
                                                                X_test, 
                                                                y_test, 
                                                                False)

        # Record the end time of training
        end_time = time.time()

        # Calculate and store the training time
        training_times[size] = end_time - start_time
        test_accuracies[size] = test_acc
        test_losses[size] = test_loss
        specificity_scores[size] = specificity
        sensitivity_scores[size] = sensitivity
        f1_scores[size] = f1

    return training_times, test_accuracies, test_losses, specificity_scores, sensitivity_scores, f1_scores

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

    # There are 1226 training images
    # There are 146 test images
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]

    from data_filtration import filter_data_rand, filter_data_strat
    from visualize_data import visualize_results

    print("There are "+str(len(X_train))+" training images and\n"+str(len(X_test))+" test images")

    # Reduce data by % randomly
    # X_train_rand, y_train_rand = filter_data_rand(X_train, y_train, 50)
    # print(len(X_train_rand))
    # print(len(y_train_rand))

    # Reduce training data by % while preserving the classes ratio
    # X_train_strat, y_train_strat = filter_data_strat(X_train, y_train, 70)
    # print(len(X_train_strat))
    # print(len(y_train_strat))

    #####
    import pickle
    import os
    training_sizes = [20, 40, 50, 60, 80]
    ####
    
    #####################################################
    ################ Run the InceptionV3 ################
    #####################################################
    # Time: 670.69 == 11.18 minutes
    # Test Loss: 1.349
    # Test Accuracy: 0.534

    # Complete data set
    # start = time.time()
    # train_inception_v3(X_train, y_train, X_test, y_test, True)
    # end = time.time()
    # print(end - start)

    # (training_times_incept1, test_accuracies_incept1, test_losses_incept1, 
    # specificity_scores_incept1, sensitivity_scores_incept1, 
    # f1_scores_incept1) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_rand,
    #                            train_inception_v3)
    
    # (training_times_incept2, test_accuracies_incept2, test_losses_incept2, 
    # specificity_scores_incept2, sensitivity_scores_incept2, 
    # f1_scores_incept2) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_strat,
    #                            train_inception_v3)
    
    # results1_v3 = {
    # 'training_times': training_times_incept1,
    # 'test_accuracies': test_accuracies_incept1,
    # 'test_losses': test_losses_incept1,
    # 'sensitivity': sensitivity_scores_incept1,
    # 'specificity': specificity_scores_incept1,
    # 'f1_scores': f1_scores_incept1
    # }

    # results2_v3 = {
    #     'training_times': training_times_incept2,
    #     'test_accuracies': test_accuracies_incept2,
    #     'test_losses': test_losses_incept2,
    #     'sensitivity': sensitivity_scores_incept2,
    #     'specificity': specificity_scores_incept2,
    #     'f1_scores': f1_scores_incept2
    # }

    # # Step 1: Combine both sets of results into a single dictionary
    # combined_results = {'results1': results1_v3, 'results2': results2_v3}

    # # Step 2: Save the combined results, overwriting the existing file
    # with open('results-inceptionv3-1.pkl', 'wb') as f:
    #     pickle.dump(combined_results, f)

    """
    Uncomment to display results from pickle file
    """
    # Get results from mobile net pickle file
    # with open('results-inceptionv3-1.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)

    # results1_v3 = loaded_results['results1']
    # results2_v3 = loaded_results['results2']

    # training_times_incept1 = results1_v3['training_times']
    # test_accuracies_incept1 = results1_v3['test_accuracies']
    # test_losses_incept1 = results1_v3['test_losses']
    # sensitivity_scores_incept1 = results1_v3['sensitivity']
    # specificity_scores_incept1 = results1_v3['specificity']
    # f1_scores_incept1 = results1_v3['f1_scores']

    # training_times_incept2 = results2_v3['training_times']
    # test_accuracies_incept2 = results2_v3['test_accuracies']
    # test_losses_incept2 = results2_v3['test_losses']
    # sensitivity_scores_incept2 = results2_v3['sensitivity']
    # specificity_scores_incept2 = results2_v3['specificity']
    # f1_scores_incept2 = results2_v3['f1_scores']

    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_incept1, 
    #                 test_accuracies_incept1, 
    #                 test_losses_incept1, 
    #                 sensitivity_scores_incept1, 
    #                 specificity_scores_incept1, 
    #                 f1_scores_incept1)
    
    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_incept2, 
    #                 test_accuracies_incept2, 
    #                 test_losses_incept2, 
    #                 sensitivity_scores_incept2, 
    #                 specificity_scores_incept2, 
    #                 f1_scores_incept2)


    ###################################################
    ################ Run the ResNet-50 ################
    ###################################################
    # Time: 770.38s == 12.84 minutes
    # Test Loss: 2.176
    # Test Accuracy: 0.685

    # start = time.time()
    # train_resnet50(X_train, y_train, X_test, y_test, False)
    # end = time.time()
    # print(end - start)

    # (training_times_res1, test_accuracies_res1, test_losses_res1, 
    # specificity_scores_res1, sensitivity_scores_res1, 
    # f1_scores_res1) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_rand,
    #                            train_resnet50)
    
    # (training_times_res2, test_accuracies_res2, test_losses_res2, 
    # specificity_scores_res2, sensitivity_scores_res2, 
    # f1_scores_res2) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_strat,
    #                            train_resnet50)
    
    # results1_rn = {
    # 'training_times': training_times_res1,
    # 'test_accuracies': test_accuracies_res1,
    # 'test_losses': test_losses_res1,
    # 'sensitivity': sensitivity_scores_res1,
    # 'specificity': specificity_scores_res1,
    # 'f1_scores': f1_scores_res1
    # }

    # results2_rn = {
    #     'training_times': training_times_res2,
    #     'test_accuracies': test_accuracies_res2,
    #     'test_losses': test_losses_res2,
    #     'sensitivity': sensitivity_scores_res2,
    #     'specificity': specificity_scores_res2,
    #     'f1_scores': f1_scores_res2
    # }

    # # Step 1: Combine both sets of results into a single dictionary
    # combined_results = {'results1': results1_rn, 'results2': results2_rn}

    # # Step 2: Save the combined results, overwriting the existing file
    # with open('results-resnet50-1.pkl', 'wb') as f:
    #     pickle.dump(combined_results, f)
    
    """
    Uncomment below to open the pickle file and get data
    """

    # # Get results from mobile net pickle file
    # with open('results-resnet50-1.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)

    # results1_rn = loaded_results['results1']
    # results2_rn = loaded_results['results2']

    # training_times_res1 = results1_rn['training_times']
    # test_accuracies_res1 = results1_rn['test_accuracies']
    # test_losses_res1 = results1_rn['test_losses']
    # sensitivity_scores_res1 = results1_rn['sensitivity']
    # specificity_scores_res1 = results1_rn['specificity']
    # f1_scores_res1 = results1_rn['f1_scores']

    # training_times_res2 = results2_rn['training_times']
    # test_accuracies_res2 = results2_rn['test_accuracies']
    # test_losses_res2 = results2_rn['test_losses']
    # sensitivity_scores_res2 = results2_rn['sensitivity']
    # specificity_scores_res2 = results2_rn['specificity']
    # f1_scores_res2 = results2_rn['f1_scores']

    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_res1, 
    #                 test_accuracies_res1, 
    #                 test_losses_res1, 
    #                 sensitivity_scores_res1, 
    #                 specificity_scores_res1, 
    #                 f1_scores_res1)
    
    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_res2, 
    #                 test_accuracies_res2, 
    #                 test_losses_res2, 
    #                 sensitivity_scores_res2, 
    #                 specificity_scores_res2, 
    #                 f1_scores_res2)

    ########################################################
    ################ Run the EfficientNetB0 ################
     #######################################################
    # Time: 135.13 == 2.25 minutes
    # Test Loss: 0.550
    # Test Accuracy: 0.747

    # start = time.time()
    # train_efficientNet(X_train, y_train, X_test, y_test, False)
    # end = time.time()
    # print(end - start)
    
    # ### SPLITS & Testing ####
    # (training_times1, test_accuracies1, test_losses1, 
    # specificity_scores1, sensitivity_scores1, 
    # f1_scores1) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_rand,
    #                            train_efficientNet)
    
    # (training_times2, test_accuracies2, test_losses2, 
    # specificity_scores2, sensitivity_scores2, 
    # f1_scores2) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_strat,
    #                            train_efficientNet)
    
    # results1 = {
    # 'training_times': training_times1,
    # 'test_accuracies': test_accuracies1,
    # 'test_losses': test_losses1,
    # 'sensitivity': sensitivity_scores1,
    # 'specificity': specificity_scores1,
    # 'f1_scores': f1_scores1
    # }

    # results2 = {
    #     'training_times': training_times2,
    #     'test_accuracies': test_accuracies2,
    #     'test_losses': test_losses2,
    #     'sensitivity': sensitivity_scores2,
    #     'specificity': specificity_scores2,
    #     'f1_scores': f1_scores2
    # }

    # Step 1: Combine both sets of results into a single dictionary
    # combined_results = {'results1': results1, 'results2': results2}

    # Step 2: Save the combined results, overwriting the existing file
    # with open('results-effnet-1.pkl', 'wb') as f:
    #     pickle.dump(combined_results, f)
    
    """
    Uncomment below to open the pickle file and get data
    """
    # # Load the results from the pickle file
    # with open('results-effnet-1.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)

    # results1 = loaded_results['results1']
    # results2 = loaded_results['results2']

    # training_times1 = results1['training_times']
    # test_accuracies1 = results1['test_accuracies']
    # test_losses1 = results1['test_losses']
    # sensitivity_scores1 = results1['sensitivity']
    # specificity_scores1 = results1['specificity']
    # f1_scores1 = results1['f1_scores']

    # training_times2 = results2['training_times']
    # test_accuracies2 = results2['test_accuracies']
    # test_losses2 = results2['test_losses']
    # sensitivity_scores2 = results2['sensitivity']
    # specificity_scores2 = results2['specificity']
    # f1_scores2 = results2['f1_scores']

    # # Use to rename file relating to efficientNet 
    # old_file_path = 'experiment_results.pkl'
    # new_file_path = 'results-effnet-1.pkl'
    # os.rename(old_file_path, new_file_path)


    # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times1, 
    #                 test_accuracies1, 
    #                 test_losses1, 
    #                 sensitivity_scores1, 
    #                 specificity_scores1, 
    #                 f1_scores1)
    
    # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times2, 
    #                 test_accuracies2, 
    #                 test_losses2, 
    #                 sensitivity_scores2, 
    #                 specificity_scores2, 
    #                 f1_scores2)
    
   
    ###################################################
    ################ Run the MobileNet ################
     ###################################################
    # Time: 166.69 == 2.78 minutes
    # Test Loss: 3.840
    # Test Accuracy: 0.644

    # start = time.time()
    # train_mobileNet(X_train, y_train, X_test, y_test, False)
    # end = time.time()
    # print(end - start)
    
    # (training_times_incept1, test_accuracies_incept1, test_losses_mbnet1, 
    # specificity_scores_mbnet1, sensitivity_scores_mbnet1, 
    # f1_scores_mbnet1) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_rand,
    #                            train_mobileNet)
    
    # (training_times_mbnet2, test_accuracies_mbnet2, test_losses_mbnet2, 
    # specificity_scores_mbnet2, sensitivity_scores_mbnet2, 
    # f1_scores_mbnet2) = run_experiment(X_train, y_train, X_test, y_test, training_sizes,
    #                            filter_data_strat,
    #                            train_mobileNet)
    
    # results1_mb = {
    # 'training_times': training_times_incept1,
    # 'test_accuracies': test_accuracies_incept1,
    # 'test_losses': test_losses_mbnet1,
    # 'sensitivity': sensitivity_scores_mbnet1,
    # 'specificity': specificity_scores_mbnet1,
    # 'f1_scores': f1_scores_mbnet1
    # }

    # results2_mb = {
    #     'training_times': training_times_mbnet2,
    #     'test_accuracies': test_accuracies_mbnet2,
    #     'test_losses': test_losses_mbnet2,
    #     'sensitivity': sensitivity_scores_mbnet2,
    #     'specificity': specificity_scores_mbnet2,
    #     'f1_scores': f1_scores_mbnet2
    # }

    # # Step 1: Combine both sets of results into a single dictionary
    # combined_results = {'results1': results1_mb, 'results2': results2_mb}

    # # Step 2: Save the combined results, overwriting the existing file
    # with open('results-mobileNet-1.pkl', 'wb') as f:
    #     pickle.dump(combined_results, f)
    
    """
    Uncomment below to open the pickle file and get data
    """

    # Get results from mobile net pickle file
    # with open('results-mobileNet-1.pkl', 'rb') as f:
    #     loaded_results = pickle.load(f)

    # results1_mb = loaded_results['results1']
    # results2_mb = loaded_results['results2']

    # training_times_incept1 = results1_mb['training_times']
    # test_accuracies_incept1 = results1_mb['test_accuracies']
    # test_losses_mbnet1 = results1_mb['test_losses']
    # sensitivity_scores_mbnet1 = results1_mb['sensitivity']
    # specificity_scores_mbnet1 = results1_mb['specificity']
    # f1_scores_mbnet1 = results1_mb['f1_scores']

    # training_times_mbnet2 = results2_mb['training_times']
    # test_accuracies_mbnet2 = results2_mb['test_accuracies']
    # test_losses_mbnet2 = results2_mb['test_losses']
    # sensitivity_scores_mbnet2 = results2_mb['sensitivity']
    # specificity_scores_mbnet2 = results2_mb['specificity']
    # f1_scores_mbnet2 = results2_mb['f1_scores']

    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_incept1, 
    #                 test_accuracies_incept1, 
    #                 test_losses_mbnet1, 
    #                 sensitivity_scores_mbnet1, 
    #                 specificity_scores_mbnet1, 
    #                 f1_scores_mbnet1)
    
    # # Pass the extracted metrics to the visualize_results function
    # visualize_results(training_sizes, 
    #                 training_times_mbnet2, 
    #                 test_accuracies_mbnet2, 
    #                 test_losses_mbnet2, 
    #                 sensitivity_scores_mbnet2, 
    #                 specificity_scores_mbnet2, 
    #                 f1_scores_mbnet2)

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################
    # Making improvements to the models:
    # X_train
    # X_test
    # y_train
    # y_test
    
    # Scale the data between 0-1
    # X_train, X_test = scale_data(X_train, X_test)

    # MobileNet
    # start = time.time()
    # (test_acc, test_loss,
    # specificity, sensitivity, f1) = train_mobileNet(X_train, 
    #                                                 y_train, 
    #                                                 X_test, 
    #                                                 y_test, 
    #                                                 False,
    #                                                 save_model=True)
    # end = time.time()
    # print(end - start)

    # print(f"Test loss: {test_loss} \nTest accuracy: {test_acc}\n")
    # print(f"Specificity: {specificity} \nSensitivity: {sensitivity} \nF1 Score: {f1}")

    # EfficientNetB0
    # start = time.time()
    # (test_acc, test_loss,
    # specificity, sensitivity, f1) = train_efficientNet(X_train, 
    #                                                    y_train, 
    #                                                    X_test, 
    #                                                    y_test, 
    #                                                    False,
    #                                                    save_model=False)
    # end = time.time()
    # print(end - start)

    # print(f"Test loss: {test_loss} \nTest accuracy: {test_acc}\n")
    # print(f"Specificity: {specificity} \nSensitivity: {sensitivity} \nF1 Score: {f1}")

    # Inception v3
    # start = time.time()
    # (test_acc, test_loss,
    # specificity, sensitivity, f1) = train_inception_v3(X_train, 
    #                                                    y_train, 
    #                                                    X_test, 
    #                                                    y_test, 
    #                                                    True,
    #                                                    save_model=True)
    # end = time.time()
    # print(end - start)

    # print(f"Test loss: {test_loss} \nTest accuracy: {test_acc}\n")
    # print(f"Specificity: {specificity} \nSensitivity: {sensitivity} \nF1 Score: {f1}")

    # ResNet-50
    # start = time.time()
    # (test_acc, test_loss,
    # specificity, sensitivity, f1) = train_resnet50(X_train, 
    #                                                y_train, 
    #                                                X_test, 
    #                                                y_test, 
    #                                                False,
    #                                                save_model=True)
    # end = time.time()
    # print(end - start)

    # print(f"Test loss: {test_loss} \nTest accuracy: {test_acc}\n")
    # print(f"Specificity: {specificity} \nSensitivity: {sensitivity} \nF1 Score: {f1}")

