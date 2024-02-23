# Jude Tear
# Feb 5th 2024

from tensorflow import keras
from keras.applications import InceptionV3, ResNet50, EfficientNetB0
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2 as cv
from data_filtration import resize_images, scale_data
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout


# Evaluation metrics / functions
def calculate_specificity_sensitivity_f1(y_true, y_pred):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    sensitivity = {}
    specificity = {}
    f1_scores = {}

    total_sum = cm.sum()

    # Calculate metrics for each class
    for i in range(len(cm)):
        # True Positives (TP) are the diagonal elements
        TP = cm[i, i]
        # False Negatives (FN) are the sum of the ith row excluding TP
        FN = sum(cm[i, :]) - TP
        # False Positives (FP) are the sum of the ith column excluding TP
        FP = sum(cm[:, i]) - TP
        # True Negatives (TN) are calculated by excluding the row and column of the current class
        TN = total_sum - (FP + FN + TP)

        # Calculate sensitivity (recall) for the current class
        sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        # Calculate specificity for the current class
        specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0
        # Calculate precision for the current class
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        # Calculate F1 score for the current class
        f1_scores[i] = 2 * (precision * sensitivity[i]) / (precision + sensitivity[i]) if (precision + sensitivity[i]) != 0 else 0

    return specificity, sensitivity, f1_scores

def evaluate_model_performance(model, X_test, y_test):
    # Predict classes with the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate metrics
    specificity, sensitivity, f1_score = calculate_specificity_sensitivity_f1(y_true_classes, y_pred_classes)
    # print(f"Specificity: {specificity}")
    # print(f"Sensitivity (Recall): {sensitivity}")
    # print(f"F1 Score: {f1_score}")

    return specificity, sensitivity, f1_score


# Models 
# Using the inception v3 model 
def train_inception_v3(X_train, y_train, X_test, y_test, resize, width=299, height=299):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(np.unique(y_train, axis=0)), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if resize:
        X_train = resize_images(X_train, width, height)
        X_test = resize_images(X_test, width, height)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[lr_reduction, early_stopping], validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    # print(f'Test loss: {test_loss}')
    # print(f'Test accuracy: {test_acc}')

    # Evaluate additional metrics
    specificity, sensitivity, f1 = evaluate_model_performance(model, X_test, y_test)
    return test_acc, test_loss, specificity, sensitivity, f1






# Using the resnet 50 model 
def train_resnet50(X_train, y_train, X_test, y_test, resize, width=224, height=224):
    
    # Implement the ResNet-50 
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # You can adjust the number of units
    predictions = Dense(3, activation='softmax')(x)  # Adjust the number 3 according to your classes

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if resize:
        X_train = resize_images(X_train, width, height)
        X_test = resize_images(X_test, width, height)

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Unfreeze some layers
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Re-compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Continue training
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    # print(f'Test loss: {loss}')
    # print(f'Test accuracy: {accuracy}')

    specificity, sensitivity, f1 = evaluate_model_performance(model, X_test, y_test)
    return test_acc, test_loss, specificity, sensitivity, f1




  

# Using the efficientNetB0
def train_efficientNet(X_train, y_train, X_test, y_test, resize, width=224, height=224):
    num_classes = 3

    model = Sequential([
    EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
    ])

    model.layers[0].trainable = False 

    if resize:
        X_train = resize_images(X_train, width, height)
        X_test = resize_images(X_test, width, height)


    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print(f'Test loss: {test_loss}')
    # print(f'Test accuracy: {test_acc}')

    specificity, sensitivity, f1 = evaluate_model_performance(model, X_test, y_test)
    return test_acc, test_loss, specificity, sensitivity, f1






# Using the mobile net
def train_mobileNet(X_train, y_train, X_test, y_test, resize, width=224, height=224):

    # Resize images if dimensions don't match expected input 
    if resize:
        X_train = resize_images(X_train, width, height)
        X_test = resize_images(X_test, width, height)

    ## Do some checking
    # min_val = X_train.min()
    # max_val = X_train.max()

    X_test = X_test.astype('float32') / 255.0

    # Import the model
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Adjustable
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)  # Adjust '3' based on number of classes

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Augment the data
    datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale images to [0, 1]
        rotation_range=10,  # Random rotations in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # Random horizontal shifts
        height_shift_range=0.2,  # Random vertical shifts
        shear_range=0.2,  # Shear transformations
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True,  # Horizontal flips
        fill_mode='nearest'  # Strategy used for filling in newly created pixels
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=32)

    model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))

    # Unfreeze the top layers of MobileNet
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Re-compile the modneel for fine-tuning
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Continue training
    model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))

    # Evaluation metrics
    test_loss, test_acc = model.evaluate(X_test, y_test)
    specificity, sensitivity, f1 = evaluate_model_performance(model, X_test, y_test)

    return test_acc, test_loss, specificity, sensitivity, f1

