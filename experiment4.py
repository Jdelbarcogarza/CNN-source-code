import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Cargar el conjunto de datos CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# network (hyper-)parameters
hidden_units = 128
dropout = 0.45
batch_size = 128
# compute the number of labels: [0, 1, ..., 9]
num_labels = len(np.unique(y_train))

# convert to one-hot vector: e.g., 2 -> [0, 0, 1, ..., 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalizar las imágenes de entrada: ajustar los valores de los píxeles de las imágenes
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Definir la arquitectura del modelo
model = Sequential()

# Convolutional layer with 32 filters, 3×3 kernel size.
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))    # Activation layer
model.add(Dropout(dropout))

# MaxPooling layer with 2×2 pool size.
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

# Flatten layer
model.add(Flatten())
model.add(Dropout(dropout))

# Dense layer with 128 units.
model.add(Dense(hidden_units))
model.add(BatchNormalization())  # Batch Normalization layer
model.add(Activation('relu'))    # Activation layer
model.add(Dropout(dropout))

# --> output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()


# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the network
history = model.fit(x_train, y_train, epochs=20, batch_size=batch_size, validation_split=0.2)



# validate the model on test dataset to determine generalization
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


import matplotlib.pyplot as plt

print(history.params)
print(history.history.keys())

_, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
axs[0].plot(history.history['loss'], marker='.', linewidth=1)
axs[0].plot(history.history['val_loss'], marker='.', linewidth=1)
axs[0].set_ylabel(r"Loss")
axs[1].plot(history.history['accuracy'], marker='.', linewidth=1)
axs[1].plot(history.history['val_accuracy'], marker='.', linewidth=1)
axs[1].set_ylabel(r"Accuracy")
axs[1].set_xlabel(r"Epoch")
axs[0].legend(["train", "validation"], loc="upper right")
plt.show()