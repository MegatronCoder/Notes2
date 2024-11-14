import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#  augment train set rescale , stretch and translate
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

# bhau batch madhe kr
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size = 32
)

# only pixel scale on test to avoid overfit
test_datagen = ImageDataGenerator(
    rescale = 1./255
)

# augment test set
test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size = 32
)

cnn = Sequential()

# add input layer
cnn.add(Input(shape=(32, 32, 3)))

#  2 layer add krte
cnn.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2,2), strides=2))
cnn.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2,2), strides=2))

# flatten the output
cnn.add(Flatten())

# add 2 fully connected layer
cnn.add(Dense(units=128, activation="relu"))
cnn.add(Dense(units=64, activation="relu"))
cnn.add(Dense(units=10, activation="softmax"))

cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "precision", "recall", "f1_score"])
history = cnn.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)
