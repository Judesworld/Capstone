# Jude Tear
# Mobile-Net file

import os
import numpy as np
import cv2 as cv
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

from evaluation import evaluate_model_performance, evaluate_model_performance_binary

class MobileNetTrainer:
    def __init__(self, num_classes, width=224, height=224, model_type='multiclass', save_model=False):
        self.width = width
        self.height = height
        self.model_type = model_type
        self.save_model = save_model
        self.num_classes = num_classes

    def resize_images(self, images, width, height):
        resized_images = np.array([cv.resize(image, (width, height)) for image in images])
        return resized_images

    def train(self, X_train, y_train, X_test, y_test, resize=False):
        print(f"\n****** Running MobileNet {self.model_type.upper()} ******")

        # Don't need to do this for default data
        if resize:
            X_train = self.resize_images(X_train, self.width, self.height)
            X_test = self.resize_images(X_test, self.width, self.height)

        # Split the data 
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


        base_model = MobileNet(weights='imagenet', 
                               include_top=False, 
                               input_shape=(self.width, self.height, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        if self.num_classes == 2 and self.model_type == 'binary':
            predictions = Dense(self.num_classes, activation='softmax')(x)  # Adjusted for binary
        else:
            predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        # Establish
        optimizer = Adam(learning_rate=0.0001)
        loss = 'categorical_crossentropy' if self.model_type == 'multiclass' else 'binary_crossentropy'
        
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=['accuracy'])
        
        # Data augmentation
        datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=10, 
                                      width_shift_range=0.2, 
                                      height_shift_range=0.2, 
                                      shear_range=0.2, 
                                      zoom_range=0.2, 
                                      horizontal_flip=True, 
                                      fill_mode='nearest')
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = datagen.flow(X_train, y_train, batch_size=32)
        test_generator = test_datagen.flow(X_test, y_test, batch_size=32)
        
        # Reduce learning rate callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                      factor=0.2, 
                                      patience=5, 
                                      min_lr=0.00001)
        
        # Initial fit
        model.fit(train_generator, 
                  epochs=10, 
                  validation_data=(X_val, y_val), 
                  callbacks=[reduce_lr])

        # Unfreeze the top layers of the base_model for fine-tuning
        # There are 86 layers in the base model
        for layer in base_model.layers[-10:]:
            layer.trainable = True
        
        # Recompile modle
        model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=['accuracy'])
        
        # Fit again
        model.fit(train_generator, 
                  epochs=10, 
                  validation_data=(X_val, y_val), 
                  callbacks=[reduce_lr])
        
        # Scale the test manually for simplicity 
        X_test = X_test.astype('float32') / 255.0
    
        # Evaluation
        test_loss, test_acc = model.evaluate(test_generator)
       
        print(f"Evaluation metrics for {self.model_type.upper()}:")
        print(f"Test Accuracy is: {test_acc}")
        print(f"Test Loss is: {test_loss}")

        # Check if model_type is 'binary' or 'multiclass' and call the appropriate evaluation function
        if self.model_type == 'binary':
            (specificity, sensitivity, f1, 
             incorrect_indicies, y_pred, y_true) = evaluate_model_performance_binary(model, X_test, y_test)
            
            # Format and print the eval metrics as %
            print(f"Specificity: {specificity * 100:.2f}%, Sensitivity: {sensitivity * 100:.2f}%, F1 Score: {f1 * 100:.2f}%")

        else:  # Multiclass 
            (specificity, sensitivity, f1, 
             incorrect_indicies, y_pred, y_true) = evaluate_model_performance(model, X_test, y_test)

            # Format and print the eval metrics as % 
            for label, metric in specificity.items():
                print(f"Class {label} - Specificity: {metric * 100:.2f}%, Sensitivity: {sensitivity[label] * 100:.2f}%, F1 Score: {f1[label] * 100:.2f}%")

    
        # Optionally save the model
        if self.save_model:
            model_save_path = os.path.join('saved_models', f'mobileNet_model_{self.model_type}.h5')
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
    
        
        return incorrect_indicies, y_pred, y_true