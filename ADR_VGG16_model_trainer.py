import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# Define the path to your dataset
dataset_path = "age_dataset"

# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift image width
    height_shift_range=0.2,  # Randomly shift image height
    shear_range=0.2,  # Randomly apply shear transformations
    zoom_range=0.2,  # Randomly zoom in on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest",  # Fill in missing pixels using the nearest value
)

# Load and preprocess training images
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Resize images to the desired input size
    batch_size=32,
    class_mode="sparse",  # Use "sparse" for regression tasks
)

# Load the VGG16 model (pre-trained on ImageNet)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Create a custom head for age estimation
head_model = base_model.output
head_model = Flatten()(head_model)
head_model = Dense(512, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="linear")(
    head_model
)  # Output layer for age regression

# Combine the base model and custom head
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(lr=0.0001), loss="mean_squared_error"
)  # Use MSE for age regression

# Train the model
model.fit(train_generator, epochs=10)

# Save the trained model using the native Keras format
model.save("advanced_age_model.keras")
