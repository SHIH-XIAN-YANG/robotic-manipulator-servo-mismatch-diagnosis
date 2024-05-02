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

from PIL import Image
import matplotlib.pyplot as plt
from model_playground import multi_channel_CNN
from tqdm import tqdm

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, images,images2,images3,images4, labels, transform=None):
        
        # self.images = images
        self.images1 = images
        self.images2 = images2
        self.images3 = images3
        self.images4 = images4

        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image1 = self.images1[idx]
        image2 = self.images2[idx]
        image3 = self.images3[idx]
        image4 = self.images4[idx]
        label = self.labels[idx]
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
            image4 = self.transform(image4)

        concatenated_image = torch.cat((image1, image2, image3, image4), dim=0)
        return concatenated_image, label

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
    train_dataset = CustomDataset((train_images1, train_images2, train_images3, train_images4), train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset((test_images1, test_images2,test_images3,test_images4), test_labels, transform=transform)
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
num_epochs = 500
batch_size = 32


# Initialize model, loss, optimizer
model = multi_channel_CNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_path = 'C:\\Users\\Samuel\\Desktop\\mismatch_dataset\\'

# Load data
train_loader, test_loader = data_split(data_path,batch_size=batch_size, transform=transform)

progress_bar = []
total_samples = len(train_loader)
n_iterations = np.ceil(total_samples / batch_size)


train_acc = []
test_acc = []
top2_train_acc = []
top2_test_acc = []
loss_epoch_C = []

#%%

param_list=[]
i=0
for param_name,param in model.named_parameters():
    i=i+1
    param_list.append(param_name)
    print(f'layer:{i}_name:{param_name}')

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    correct_test, total_test = 0, 0
    top2_correct_train, top2_correct_test = 0,0
    train_loss_C = 0.0


    for i, (inputs, labels) in enumerate(train_loader):
        if (i+1) % (total_samples/100) == 0:
            progress_bar.append("=")
            print(f' ||epoch {epoch+1}/{num_epochs}, step {i+1}/{total_samples}',end='\r')

            #print(f'epoch {epoch+1}/{epochs}|',end="")
            for idx, progress in enumerate(progress_bar):
                print(progress,end="")
        inputs = [input.to(device) for input in inputs]
        labels = labels.to(device)
        optimizer.zero_grad()
        # print(type(inputs))
        # print(len(inputs))
        outputs = model(inputs)
        # print(outputs.shape)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        _, predicted_top2 = torch.topk(outputs.data, 2, dim=1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        top2_correct_train += torch.sum(torch.eq(predicted_top2[:, 0], labels) | torch.eq(predicted_top2[:, 1], labels)).item()


        train_loss_C += loss.item()
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    print(f'Training epoch: {epoch + 1}/{num_epochs} / loss_C: {train_loss_C/len(train_loader)} | acc: {correct_train / total_train} | top 2 acc: {top2_correct_train/total_train}')

    progress_bar = []
                    

    # Testing loop
    model.eval()
  
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_top2 = torch.topk(outputs.data, 2, dim=1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            top2_correct_test += torch.sum(torch.eq(predicted_top2[:, 0], labels) | torch.eq(predicted_top2[:, 1], labels)).item()


    print(f'Testing acc : {correct_test / total_test} | Top 2 Test acc: {top2_correct_test / total_test}')


    train_acc.append(100 * (correct_train / total_train)) # training accuracy
    test_acc.append(100 * (correct_test / total_test))    # testing accuracy
    loss_epoch_C.append((train_loss_C / len(train_loader)))            # loss 
    top2_train_acc.append(100*(top2_correct_train/total_train)) # training accuracy
    top2_test_acc.append(100*(top2_correct_test / total_test))

# %%

plt.figure()
plt.plot(list(range(num_epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.grid(True)
plt.show()
plt.savefig('parallel_cnn_loss.png')

plt.figure()
plt.plot(list(range(num_epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(num_epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.grid(True)
plt.show()
plt.savefig('parallel_cnn_acc.png')

plt.figure()
plt.plot(list(range(num_epochs)), top2_train_acc)    # plot your training accuracy
plt.plot(list(range(num_epochs)), top2_test_acc)     # plot your testing accuracy
plt.title('Top 2 acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'val acc'], loc = 'upper left')
plt.grid(True)
plt.show()
plt.savefig('parallel_cnn_top2_acc.png')
# %%
