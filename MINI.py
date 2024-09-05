import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# Define paths
train_dir = '/Users/Saibharat/Desktop/Mini Project/Dataset/dataset'

# Parameters
img_height, img_width = 224, 224
batch_size = 32
num_classes = 4  # Assuming each folder represents a class
epochs = 10


# Data Preparation
def create_data_generators(train_dir, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


train_generator, validation_generator = create_data_generators(train_dir, img_height, img_width, batch_size)
print(len(train_generator))
print(len(validation_generator))


# Model Creation
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


model = create_model((img_height, img_width, 3), num_classes)
model.summary()


# Model Training
def train_model(model, train_generator, validation_generator, epochs):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    return history


history = train_model(model, train_generator, validation_generator, epochs)


# Prediction
def predict_image(model, img_path, img_height, img_width, class_indices):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]

    return predicted_label


# Load the best model
model = load_model('best_model.keras')

# Example usage:
img_path = '/Users/Saibharat/Desktop/Mini Project/Dataset/image.jpeg'
predicted_label = predict_image(model, img_path, img_height, img_width, train_generator.class_indices)
print(f'This shot is classified as: {predicted_label}')
