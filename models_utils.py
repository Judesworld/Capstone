# Jude Tear
# Feb 5th 2024

from tensorflow import keras
from keras.applications import InceptionV3, ResNet50, EfficientNetB0
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
import cv2 as cv
import numpy as np

def resize_images(data, width, height):
    resized_data = np.empty((data.shape[0], height, width, data.shape[3]))
    for i, img in enumerate(data):
        resized_data[i] = cv.resize(img, (width, height))
    return resized_data

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

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

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

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')
    return 

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
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_acc}')

    return

def train_mobileNet(X_train, y_train, X_test, y_test, resize, width=224, height=224):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Adjustable based on your dataset and complexity
    predictions = Dense(3, activation='softmax')(x)  # Adjust '3' based on your number of classes

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    if resize:
        X_train = resize_images(X_train, width, height)
        X_test = resize_images(X_test, width, height)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Unfreeze the top layers of MobileNet
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Re-compile the model for fine-tuning
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Continue training
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')
    return 