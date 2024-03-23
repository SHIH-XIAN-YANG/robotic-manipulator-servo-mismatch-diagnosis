### Neural network

import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History

import matplotlib.pyplot as plt
NN_data_dir = 'NN_data/'
bandwidth = np.loadtxt(NN_data_dir+'Bandwidth.txt', delimiter=',')
contouring_err = np.loadtxt(NN_data_dir+'contouring_err.txt', delimiter=',')
labels = np.loadtxt(NN_data_dir+'labels.txt', delimiter=',')
tracking_err = np.loadtxt(NN_data_dir+'tracking_err.txt', delimiter=',')
link_gain = np.loadtxt(NN_data_dir+'link_gain.txt', delimiter=',')


input_size = contouring_err.shape[1] #length of row in array contouring_err
output_size = bandwidth.shape[1]

print(link_gain[0])


print(f"Input size: {input_size} || Output size: {output_size}")
epochs = 200
batch_size = 4

X_train, X_val, y_train, y_val = train_test_split(tracking_err, labels, test_size=0.2, random_state=42)


# Construct NN model
model = Sequential()

# Add the first hidden layer
first_layer_node_number = 128
model.add(Dense(first_layer_node_number, activation='relu', input_shape=(input_size,)))
model.add(Dropout(0.25))

# Add the second hidden layer
second_layer_node_number = 128
model.add(Dense(second_layer_node_number, activation='relu'))
model.add(Dropout(0.2))

# Add the third hidden layer
third_layer_node_number = 64
model.add(Dense(second_layer_node_number, activation='relu'))
#model.add(Dropout(0.2))

# Add the third hidden layer
fourth_layer_node_number = 32
model.add(Dense(second_layer_node_number, activation='relu'))

# Add the output layer
model.add(Dense(output_size, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_val, y_val), batch_size=batch_size,verbose=1)


#Display the training progress
print(history.history)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
training_acc =history.history['accuracy']
val_acc = history.history['val_accuracy']

# plot the result
plt.plot(training_loss, label='train_loss')
plt.plot(validation_loss, label='val_loss')
plt.plot(training_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.grid(True)
plt.title('Neural netork result')
plt.legend()
plt.savefig(f'trained_result/NN/arch_{first_layer_node_number}_{second_layer_node_number}_{third_layer_node_number}_{fourth_layer_node_number}_result.png')
plt.show()


# Save the model
model.save(f'saved_model/NN/arch_{first_layer_node_number}_{second_layer_node_number}_{third_layer_node_number}_{fourth_layer_node_number}_train_acc_{training_acc[-1]:.3f}_val_acc_{val_acc[-1]:.3f}.h5')
