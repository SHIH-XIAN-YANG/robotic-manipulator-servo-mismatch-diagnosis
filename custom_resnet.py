#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import SGD, Adam, AdamW
import pymysql
import os
import numpy as np
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt
from model_playground import CustomResNet
from model_playground import CustomDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
# Define custom dataset
# class CustomDataset(Dataset):
#     def __init__(self, images1, images2,images3,images4, labels, transform):
#         self.images1 = images1
#         self.images2 = images2
#         self.images3 = images3
#         self.images4 = images4

#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image1 = self.images1[idx]
#         image2 = self.images2[idx]
#         image3 = self.images3[idx]
#         image4 = self.images4[idx]
#         label = self.labels[idx]
#         if self.transform:
#             image1 = self.transform(image1)
#             image2 = self.transform(image2)
#             image3 = self.transform(image3)
#             image4 = self.transform(image4)

#         concatenated_image = torch.cat((image1, image2, image3, image4), dim=0)
#         return concatenated_image, label

def data_load(data_path):
    # connect to databse
    connction = pymysql.connect(
        host = "localhost",
        user="root",
        port=3306,
        password="Sam512011",
        database="bw_mismatch_db"
    )

    cursor = connction.cursor()
    # print(bandwidth)
    sql = "SELECT min_bandwidth FROM bw_mismatch_data"
    cursor.execute(sql)
    min_bandwidth = np.array([])
    data = cursor.fetchall()
    all_labels = []
    # for idx, row in enumerate(data):
    #     all_labels.append(row[0])

    # Define paths for images and labels (modify based on your data structure)
    images1_path = os.path.join(data_path, "circular_error")
    images2_path = os.path.join(data_path, "orientation_contour_error")
    images3_path = os.path.join(data_path, "phase_delay")
    images4_path = os.path.join(data_path, "z_error")

    # Load image paths (modify based on your file naming convention)
    all_images1 = [os.path.join(images1_path, f) for f in os.listdir(images1_path)]
    all_images2 = [os.path.join(images2_path, f) for f in os.listdir(images2_path)]
    all_images3 = [os.path.join(images3_path, f) for f in os.listdir(images3_path)]
    all_images4 = [os.path.join(images4_path, f) for f in os.listdir(images4_path)]

    for filename in all_images1:
        id = int(os.path.splitext(os.path.basename(filename))[0])
        sql = "SELECT min_bandwidth FROM bw_mismatch_data WHERE id=%s"
        cursor.execute(sql,(id, ))
        data = cursor.fetchall()
        # print(id, data)
        all_labels.append(data[0][0])
        # print(id, data[0][0])

    # print(all_labels)
    # print(len(all_labels))

    return all_images1, all_images2, all_images3, all_images4, all_labels

# Split data and create loaders
def data_split(data_path, split_ratio=0.8, batch_size=32, transform=None):
    # Load your data from 'data_path' here (assuming images and labels are accessible)
    # ...
    all_images1, all_images2,all_images3,all_images4, all_labels = data_load(data_path)


    # Split data into train and test sets
    num_samples = len(all_labels)
    train_size = int(num_samples * split_ratio)
    train_images1, train_images2, train_images3,train_images4, train_labels = all_images1[:train_size], all_images2[:train_size],all_images3[:train_size],all_images4[:train_size] , all_labels[:train_size]
    test_images1, test_images2,test_images3,test_images4, test_labels = all_images1[train_size:], all_images2[train_size:],all_images3[train_size:], all_images4[train_size:], all_labels[train_size:]

    # Load and pre-process images (modify based on your needs)
    def load_and_preprocess(image_path):
        # print(image_path)
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format

        return image
    
    train_images1 = [load_and_preprocess(path) for path in train_images1]
    train_images2 = [load_and_preprocess(path) for path in train_images2]
    train_images3 = [load_and_preprocess(path) for path in train_images3]
    train_images4 = [load_and_preprocess(path) for path in train_images4]
    test_images1 = [load_and_preprocess(path) for path in test_images1]
    test_images2 = [load_and_preprocess(path) for path in test_images2]
    test_images3 = [load_and_preprocess(path) for path in test_images3]
    test_images4 = [load_and_preprocess(path) for path in test_images4]

    # Create datasets and dataloaders
    train_dataset = CustomDataset([train_images1, train_images2, train_images3, train_images4], labels=train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset([test_images1, test_images2,test_images3,test_images4], labels=test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Define transformation (if needed)
transform = transforms.Compose([
    transforms.ToTensor()
    # Add more transforms as needed
])

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Define training parameters
learning_rate = 0.001
num_epochs = 2
batch_size = 64


# Initialize model, loss, optimizer
model = CustomResNet(input_channels=4*3).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_path = 'C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\'

# Load data
train_loader, test_loader = data_split(data_path,batch_size=batch_size, transform=transform)

total_samples = len(train_loader)
n_iterations = np.ceil(total_samples / batch_size)


train_acc = []
test_acc = []
top2_train_acc = []
top2_test_acc = []
loss_epoch_C = []
train_precision_list, train_recall_list, train_f1_list = [], [], []
test_precision_list, test_recall_list, test_f1_list = [], [], []

#%%


early_stop_count = 10

param_list=[]
i=0
for param_name,param in model.named_parameters():
    i=i+1
    param_list.append(param_name)
    print(f'layer:{i}_name:{param_name}')

for param_name,param in model.named_parameters():
    if param_name not in param_list[len(param_list)-10:]:
        param.requires_grad=False

# Training loop
for epoch in range(num_epochs):
    if early_stop_count==0:
        print(f'early stop at epoch {epoch}')
        break
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    correct_test, total_test = 0, 0
    top2_correct_train, top2_correct_test = 0,0
    train_loss_C = 0.0

    all_train_labels = []
    all_train_predictions = []

    if epoch == 20:
        learning_rate=0.0003
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-10:len(param_list)-2]:
                param.requires_grad=True
    elif epoch ==50:
        learning_rate=0.0002
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-31:len(param_list)-10]:
                param.requires_grad=True
    elif epoch ==150:
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-58:len(param_list)-31]:
                param.requires_grad=True
    elif epoch ==250:
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-88:len(param_list)-58]:
                param.requires_grad=True
    elif epoch ==400:
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-106:len(param_list)-88]:
                param.requires_grad=True
    elif epoch ==450:
        learning_rate=0.0001
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-136:len(param_list)-106]:
                param.requires_grad=True
    elif epoch ==550:
        for param_name,param in model.named_parameters():
            if param_name in param_list[len(param_list)-161:len(param_list)-136]:
                param.requires_grad=True
                
    parameter=filter(lambda p:p.requires_grad,model.parameters())
    #optimizer=SGD(parameter, lr=lr, momentum=0.9, nesterov = True)
    #optimizer = Adam(parameter, lr=lr,eps=1e-08)
    optimizer_C=AdamW(parameter, lr=learning_rate, amsgrad=True)

    for i, (inputs, labels) in tqdm(enumerate(train_loader)):

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_C.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_C.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        _, predicted_top2 = torch.topk(outputs.data, 2, dim=1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        top2_correct_train += torch.sum(torch.eq(predicted_top2[:, 0], labels) | torch.eq(predicted_top2[:, 1], labels)).item()

        train_loss_C += loss.item()

        all_train_labels.extend(labels.cpu().numpy())
        all_train_predictions.extend(predicted.cpu().numpy())

    if correct_train/total_train==1.0:
        print(f'acc {correct_train/total_train} already high')
        early_stop_count = early_stop_count-1
        
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    print(f'Training epoch: {epoch + 1}/{num_epochs} / loss_C: {train_loss_C/len(train_loader)} | acc: {correct_train / total_train} | top 2 acc: {top2_correct_train/total_train}')
                    

    # Testing loop
    model.eval()

    all_test_labels = []
    all_test_predictions = []
  
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_top2 = torch.topk(outputs.data, 2, dim=1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            top2_correct_test += torch.sum(torch.eq(predicted_top2[:, 0], labels) | torch.eq(predicted_top2[:, 1], labels)).item()

            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())


    train_precision = precision_score(all_train_labels, all_train_predictions, average='macro', zero_division=0)
    train_recall = recall_score(all_train_labels, all_train_predictions, average='macro', zero_division=0)
    train_f1 = f1_score(all_train_labels, all_train_predictions, average='macro', zero_division=0)

    test_precision = precision_score(all_test_labels, all_test_predictions, average='macro', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_predictions, average='macro', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_predictions, average='macro', zero_division=0)

    print(f'Training Precision: {train_precision} | Training Recall: {train_recall} | Training F1-Score: {train_f1}')
    print(f'Testing Precision: {test_precision} | Testing Recall: {test_recall} | Testing F1-Score: {test_f1}')
    # print(f'Testing acc : {correct_test / total_test} | Top 2 Test acc: {top2_correct_test / total_test}')


    train_acc.append(100 * (correct_train / total_train)) # training accuracy
    test_acc.append(100 * (correct_test / total_test))    # testing accuracy
    loss_epoch_C.append((train_loss_C / len(train_loader)))            # loss 
    top2_train_acc.append(100*(top2_correct_train/total_train)) # training accuracy
    top2_test_acc.append(100*(top2_correct_test / total_test))

    train_precision_list.append(train_precision)
    train_recall_list.append(train_recall)
    train_f1_list.append(train_f1)

    test_precision_list.append(test_precision)
    test_recall_list.append(test_recall)
    test_f1_list.append(test_f1)

# %%

test_diff = np.diff(test_acc)
for i,diff in enumerate(test_diff):
    if diff<0 and abs(diff)<0.4: # overfitting occured
        loss_epoch_C = loss_epoch_C[0:i-1]
        train_acc = train_acc[0:i-1]
        test_acc = test_acc[0:i-1]
        top2_train_acc = top2_train_acc[0:i-1]
        top2_test_acc = top2_test_acc[0:i-1]
        num_epochs = i-1
        


# Get the current date and time
current_datetime = datetime.now()

# Extract year, month, day, hour, and minute components
year = current_datetime.year
month = current_datetime.month
day = current_datetime.day
hour = current_datetime.hour
minute = current_datetime.minute


# Save the model (optional)
plt.figure()
plt.plot(list(range(num_epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.grid(True)

plt.savefig(f'resNet_{month}_{day}_{hour}_{minute}_loss.png')

plt.figure()
plt.plot(list(range(num_epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(num_epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.grid(True)
plt.savefig(f'resNet_{month}_{day}_{hour}_{minute}_top1_accuracy.png')

plt.figure()
plt.plot(list(range(num_epochs)), top2_train_acc)    # plot your training accuracy
plt.plot(list(range(num_epochs)), top2_test_acc)     # plot your testing accuracy
plt.title('Top 2 acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'val acc'], loc = 'upper left')
plt.grid(True)
plt.savefig(f'resNet_{month}_{day}_{hour}_{minute}_top2_accuracy.png')

plt.figure()
plt.plot(list(range(num_epochs)), train_f1_list, 'b', label='Training F1 Score')
plt.plot(list(range(num_epochs)), test_f1_list, 'r', label='Testing F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'resNet_{month}_{day}_{hour}_{minute}_F1_score.png')
plt.show()
