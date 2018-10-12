# Import modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


##############################################
# Preparation of input dataset
##############################################

# use the Fashion dataset provided by Keras
# Dataset of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# create a python list of the 10 fashion categories (class labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Shapes of training set
print("Training set images shape: {shape}".format(shape=train_images.shape))
print("Training set labels shape: {shape}".format(shape=train_labels.shape))


# Shapes of test set
print("Test set images shape: {shape}".format(shape=test_images.shape))
print("Test set labels shape: {shape}".format(shape=test_labels.shape))


# pixel values are 0:255, normalize to range of 0:1
train_images = train_images / 255.0
test_images = test_images / 255.0

##############################################
# Keras Sequential model
##############################################
# 1st layer flattens 28x28 array into 784 pixel vector
# 2nd layer is fully-connected 128 node layer with ReLu activation
# output layer is fully-connected 10-node softmax layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# print a summary of the model
print(model.summary())



##############################################
# Compile model
##############################################
# Adam optimizer to change weights & biases
# Loss function is sparse categorical crossentropy
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


##############################################
# Train model with training set
##############################################
# the keras fit function trains the model and returns a 
# history object whose .history attribute is a python 
# dictionary of training and loss metrics

# train the model
history = model.fit(train_images, train_labels, batch_size=32, epochs=5)


# plot history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


##############################################
# Evaluate model accuracy with test set
##############################################
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


##############################################
# Used trained model to make predictions
##############################################

print("\nLet's make some predictions with the trained model..")
predictions = model.predict(test_images)

# each prediction is an array of 10 values
# the max of the 10 values is the model's 
# highest "confidence" classification

# use numpy argmax function to get highest of the set of 10

print("Predict 1st sample in the test set is: {pred}".format(pred=class_names[np.argmax(predictions[0])]))
print("The 1st sample in test set actually is: {actual}".format(actual=class_names[test_labels[0]]))

print("Predict 5th sample in the test set is: {pred}".format(pred=class_names[np.argmax(predictions[4])]))
print("The 5th sample in test set actually is: {actual}".format(actual=class_names[test_labels[4]]))





