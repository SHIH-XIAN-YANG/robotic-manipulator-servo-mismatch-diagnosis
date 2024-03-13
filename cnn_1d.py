#%%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import json
import numpy as np
import pymysql
import time

from model_playground import *
    

#%%

# connect to databse
connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database="bw_mismatch_db"
)

cursor = connction.cursor()


"""
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|  id |  Gain | BW | C_error | min_bandwidth | t_errX | t_errY | t_errZ | tracking_err_pitch | tracking_err_roll | tracking_err_yaw | contour_err_img_path | ori_contour_err_img_path|
|-----|-------|----|---------|---------------|--------|--------|--------|--------------------|-------------------|------------------|----------------------|-------------------------|
...

"""
# sql = "SELECT Gain FROM bw_mismatch_data;"
# cursor.execute(sql)
# gain = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(json.loads(row[0]))
#     gain = np.append(gain, json.loads(row[0]),axis=0)
# gain = gain.reshape((-1,len(json.loads(row[0]))))


# sql = "SELECT Bandwidth FROM bw_mismatch_data;"
# cursor.execute(sql)
# bandwidth = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(json.loads(row[0]))
#     bandwidth = np.append(bandwidth, json.loads(row[0]))
# bandwidth = bandwidth.reshape((-1,len(json.loads(row[0]))))

# print(bandwidth)
sql = "SELECT min_bandwidth FROM bw_mismatch_data;"
cursor.execute(sql)
min_bandwidth = np.array([])
data = cursor.fetchall()
for idx, row in enumerate(data):
    # print(row[0])
    arr = [0]*6
    arr[row[0]] = 1
    min_bandwidth = np.append(min_bandwidth,[arr])
min_bandwidth = min_bandwidth.reshape((-1, len(arr)))

# sql = "SELECT contour_err FROM bw_mismatch_data;"
# cursor.execute(sql)
# contour_err = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(len(json.loads(row[0])))
#     contour_err = np.append(contour_err, json.loads(row[0]))
# # print(row)
# contour_err = contour_err.reshape((-1, len(json.loads(row[0]))))

# sql = "SELECT ori_contour_err FROM bw_mismatch_data;"
# cursor.execute(sql)
# ori_contour_err = np.array([])
# data = cursor.fetchall()
# for idx, row in enumerate(data):
#     # print(len(json.loads(row[0])))
#     ori_contour_err = np.append(ori_contour_err, json.loads(row[0]))
# # print(row)
# ori_contour_err = ori_contour_err.reshape((-1, len(json.loads(row[0]))))

sql = "SELECT id, min_bandwidth, tracking_err_j1, tracking_err_j2, tracking_err_j3, tracking_err_j4, tracking_err_j5, tracking_err_j6 FROM bw_mismatch_joints_data;"
cursor.execute(sql)
tracking_err_j2 = np.array([])
data = cursor.fetchall()

min_bandwidth = []
tracking_err_joints = [[] for _ in range(6)]

for _, row in enumerate(data):
    min_bandwidth.append(row[1])
    for i in range(6):
        tracking_err_joints[i].append(json.loads(row[i+2]))


inputs = np.array(tracking_err_joints)
inputs = inputs.transpose((1,0,2))

outputs = []    
for i, bw in enumerate(min_bandwidth):
    arr = [0]*6
    arr[bw] = 1
    outputs.append(arr)
outputs = np.array(outputs)


"""
input data: contour_err
output(predict result): min_bandwidth

"""

# contour_err = np.stack((contour_err, ori_contour_err), axis=1)

# print(contour_err.shape)
#%%

epochs = 700
batch_size = 1024


input_shape = (inputs[0,:,:].shape) #length of row in array contouring_err
output_shape = (outputs.shape[1])

print(input_shape, output_shape)
# Create dataset using a custom class (optional, but recommended for flexibility)
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
# Split data into train and test sets
train_size = int(0.8 * inputs.shape[0])
test_size = inputs.shape[0] - train_size

print(f'train size : {train_size}')
print(f'test size : {test_size}')

# contour_err = torch.tensor(contour_err, dtype=torch.float32)
# min_bandwidth = torch.tensor(min_bandwidth, dtype=torch.long)

# Create the combined dataset
dataset = CustomDataset(inputs, outputs)

# Perform train-test split using random_split for better shuffling
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Configure dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for testing


# Define model and optimizer
# Define device (use GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Define loss function
criterion = nn.CrossEntropyLoss()


# Create the network and optimizer
model = CNN1D(input_dim=input_shape[0], output_dim=output_shape)
summary(model,input_shape)
# print(model)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

train_acc = []
test_acc = []
loss_epoch_C = []
top2_train_acc = []
top2_test_acc = []

#%%
# Train the model
for epoch in range(epochs):
    correct_train, total_train = 0, 0
    correct_test, total_test = 0, 0
    top2_correct_train, top2_correct_test = 0,0
    train_loss_C = 0

    for data, target in train_dataloader:
        # if torch.is_tensor(data) and data.dtype == torch.double:
        #     data = data.float()
            # target  = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()


        _, predicted = torch.max(output.data, 1)
        _, predicted_top2 = torch.topk(output.data, 2, dim=1)
        target = torch.argmax(target, dim=1)
        
        # print(predicted, target)

        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
        top2_correct_train += torch.sum(torch.eq(predicted_top2[:, 0], target) | torch.eq(predicted_top2[:, 1], target)).item()

        train_loss_C += loss.item()

    print(f'Training epoch: {epoch + 1}/{epochs} / loss_C: {train_loss_C/len(train_dataloader)} | acc: {correct_train / total_train} | top 2 acc: {top2_correct_train/total_train}')


    # Evaluate on test set (optional)
    with torch.no_grad():
        correct = 0
        # total = 0
        for data, target in test_dataloader:
            
            # if torch.is_tensor(data) and data.dtype == torch.double:
            #     data = data.float()
                # target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.data.shape)
            total_test += target.size(0)
            predicted = torch.argmax(output.data, dim=1)
            _, predicted_top2 = torch.topk(output.data, 2, dim=1)
            target = torch.argmax(target, dim=1)
            
            # print(f'predice t= {predicted}')
            
            # print(predicted)
            # print(target)
            correct_test += (predicted == target).sum().item()
            top2_correct_test += torch.sum(torch.eq(predicted_top2[:, 0], target) | torch.eq(predicted_top2[:, 1], target)).item()
        train_acc.append(100 * correct_train / total_train)
        test_acc.append(100 * correct_test / total_test)
        loss_epoch_C.append(train_loss_C / len(train_dataloader))
        top2_train_acc.append(100*(top2_correct_train/total_train)) # training accuracy
        top2_test_acc.append(100*(top2_correct_test / total_test))
        print(f'Testing acc : {correct_test / total_test} | Top 2 Test acc: {top2_correct_test / total_test}')

# Save the model (optional)
torch.save(model.state_dict(), "model.pth")
plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(list(range(epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(list(range(epochs)), top2_train_acc)    # plot your training accuracy
plt.plot(list(range(epochs)), top2_test_acc)     # plot your testing accuracy
plt.title('Top 2 acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'val acc'], loc = 'upper left')
plt.grid(True)
plt.savefig('top2_accuracy.png')
# Save the model
# model.save(f'saved_model/NN/arch_{first_layer_node_number}_{second_layer_node_number}_train_acc_{training_acc[-1]:.3f}_val_acc_{val_acc[-1]:.3f}.h5')
# %%

cursor.close()
connction.close()

