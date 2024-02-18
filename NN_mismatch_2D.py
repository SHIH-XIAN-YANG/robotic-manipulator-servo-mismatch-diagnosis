### Neural network
#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
from tensorflow.python.client import device_lib
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys

import matplotlib.pyplot as plt
NN_data_dir = 'NN_data/train_data/'


#%%
# bandwidth = np.loadtxt(NN_data_dir+'Bandwidth.txt', delimiter=' ')
# contouring_err = np.loadtxt(NN_data_dir+'contour_err.txt', delimiter=' ')
labels = np.loadtxt(NN_data_dir+'labels.txt', delimiter=' ')
# tracking_err = np.loadtxt(NN_data_dir+'tracking_err.txt', delimiter=' ')
tracking_err_x = np.loadtxt(NN_data_dir+'tracking_err_x.txt', delimiter=' ')
tracking_err_y = np.loadtxt(NN_data_dir+'tracking_err_y.txt', delimiter=' ')
tracking_err_z = np.loadtxt(NN_data_dir+'tracking_err_z.txt', delimiter=' ')
tracking_err_Psi = np.loadtxt(NN_data_dir+'tracking_err_Psi.txt', delimiter=' ')
tracking_err_Phi = np.loadtxt(NN_data_dir+'tracking_err_Phi.txt', delimiter=' ')
tracking_err_theta = np.loadtxt(NN_data_dir+'tracking_err_theta.txt', delimiter=' ')
# link_gain = np.loadtxt(NN_data_dir+'links_gain.txt', delimiter=' ')

input_data = np.stack((tracking_err_x, tracking_err_y,tracking_err_Psi,tracking_err_Phi,tracking_err_theta, tracking_err_z), axis=1)
#%%
input_shape = input_data.shape[1:3] #length of row in array contouring_err
output_shape = labels.shape[1]


print(input_shape)
print(output_shape)

epochs = 500
batch_size = 8

X_train, X_val, y_train, y_val = train_test_split(input_data, labels, test_size=0.2, random_state=42)

first_layer_node_number = 128
second_layer_node_number = 64


#%%
# Construct NN model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(first_layer_node_number, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(second_layer_node_number, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(output_shape, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

#%%
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

train_gen = DataGenerator(X_train, y_train, batch_size=batch_size)
val_gen = DataGenerator(X_val, y_val, batch_size=batch_size)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(device_lib.list_local_devices())
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
#%%
history = model.fit(train_gen, 
                    epochs=epochs,
                    validation_data=val_gen, 
                    batch_size=batch_size,verbose=1,
                    )

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
plt.savefig(f'trained_result/NN/arch_{first_layer_node_number}_{second_layer_node_number}_result.png')
plt.show()


# Save the model
model.save(f'saved_model/NN/arch_{first_layer_node_number}_{second_layer_node_number}_train_acc_{training_acc[-1]:.3f}_val_acc_{val_acc[-1]:.3f}.h5')
# %%
