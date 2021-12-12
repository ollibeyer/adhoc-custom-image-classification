import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Load config
CONFIG_FILE = "config.json"
f = open(CONFIG_FILE)
config = json.load(f)
f.close()

BATCH_SIZE      = config['batch_size']
FREEZE          = 1.0-config['unfreeze']
EPOCHS          = config['epochs']
LEARNING_RATE   = config['learning_rate']

IMG_SIZE = (224, 224)   # MobileNetV2
DATASET_DIR = "dataset"
LABEL_FILE = "labels.txt"

train_dataset = image_dataset_from_directory(DATASET_DIR,
                                            subset="training",
                                            labels='inferred',
                                            label_mode='categorical',
                                            shuffle=True,
                                            seed=123,
                                            validation_split=0.8,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
val_dataset = image_dataset_from_directory(DATASET_DIR,
                                            subset="validation",
                                            labels='inferred',
                                            label_mode='categorical',
                                            shuffle=True,
                                            seed=123,
                                            validation_split=0.8,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)
NUM_CLASSES = len(train_dataset.class_names)
with open(LABEL_FILE, 'w') as file:
    for line in train_dataset.class_names:
        file.write(line + "\n")
    file.close()

# Configure dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Prepare image augmentation
img_augmentation = Sequential(
    [
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(width_factor=0.1, height_factor=0.1),
        layers.RandomContrast(0.1),
    ],
    name="img_augmentation",
)

# Build and configure model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = True

# Freeze the X layers (and batch normalized) of base model to enable fine tuning.
num_layers = len(base_model.layers)
freeze_layers = round(num_layers * FREEZE)
for layer in base_model.layers[:freeze_layers]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
for layer in base_model.layers[freeze_layers:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = layers.Input(shape=IMG_SHAPE)
x = img_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
x = layers.BatchNormalization()(x)
top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs, name="MobileNet")

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

model.summary()

# Define early stopping callback
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset,
                    callbacks=[early_stop_cb])

# Save model
model.save("model.h5", save_format='h5')

# Plot results
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(history)