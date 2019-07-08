
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
num_classes = 10
labels = [ "T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


t1 = time.time()
# let's print the shape before we reshape and normalize
print(f'X_train shape {X_train.shape} ')
print(f'y_train shape {y_train.shape} ')
print(f'X_test shape  {X_test.shape} ')
print(f'y_test shape  {y_test.shape} ')

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# print the final input shape ready for training
print(f'Train matrix shape   {X_train.shape} ')
print(f'Test matrix shape    {X_test.shape} ')
t2 = time.time()
print(f'Preprocessing took {t2 - t1} seconds')

t1 = time.time()
# Create an object of class Sequential to hold modules
model = Sequential()

# First hidden layer: create 784 input neurons and 512 neurons in hidden layer
model.add(Dense(512, input_shape=(784,), activation='relu'))
model.add(Dropout(0.2))

# Second hidden layer, 512 neurons
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Output layer : 10 output neurons as there are 10 categories of clothes to choose from 
model.add(Dense(10, activation='softmax'))


print("Training network ...")
model.compile(loss='categorical_crossentropy', optimizer='adam')

print("Fitting data to model")
model.fit(X_train, Y_train, batch_size=128, epochs=5, 
          verbose=2, validation_data=(X_test, Y_test))
t2 = time.time()
print(f'Training model took {(t2 - t1):.4f} seconds ')

n = 0
t1 = time.time()
print("Making predictions...")
img = np.array(X_test[n][np.newaxis,:])
preds = int(model.predict_classes(img))
acc_preds = model.predict(img)
print(f'Me thinks me saw a : {labels[preds]} as number {preds}  ')
index = 0
for i in acc_preds[0]:
    print(f'Accuracy : {index} ->  {i:.6f}')
    index += 1
t2 = time.time()
print(f'Predictions took {(t2 - t1):.4f} seconds')



