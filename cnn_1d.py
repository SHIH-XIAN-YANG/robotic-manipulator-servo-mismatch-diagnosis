
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import pymysql
import time
from tqdm import tqdm
from datetime import datetime
import random

from model_playground import *
    

#%%

# connect to databse

db = "bw_mismatch_db"
table_name = "joint_tracking_err_table_new"
print(f"Connect to database {db}")
connction = pymysql.connect(
    host = "localhost",
    user="root",
    port=3306,
    password="Sam512011",
    database=db
)

cursor = connction.cursor()

sql = f"SELECT id, min_bandwidth, tracking_err_j1, tracking_err_j2, tracking_err_j3, tracking_err_j4, tracking_err_j5, tracking_err_j6 FROM {table_name};"
cursor.execute(sql)
data = cursor.fetchall()

min_bandwidth = []
train_data = [[] for _ in range(6)]

print("load data...")
for _, row in tqdm(enumerate(data), total=len(data)):
    if(row[1] == 3): #ignore half of data from class "3" to avoid inbalance data
        if random.randint(1,10)%2:
            continue
    min_bandwidth.append(row[1])
    for i in range(6):
        train_data[i].append(json.loads(row[i+2]))

inputs = np.array(train_data)
inputs = inputs.transpose((1,0,2))
print('train data shape:', inputs.shape)

outputs = []    
for i, bw in enumerate(min_bandwidth):
    arr = [0]*6
    arr[bw] = 1
    outputs.append(arr)
outputs = np.array(outputs)

print('output data shape: ',outputs.shape)

# plt.show()

# print(contour_err.shape)
#%%

epochs = 1000
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

# Create the combined dataset
dataset = CustomDataset(inputs, outputs)

# Perform train-test split using random_split for better shuffling
# X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(
    inputs, outputs, test_size=0.3, stratify=outputs, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f'Train size: {X_train.shape[0]}')
print(f'Validation size: {X_val.shape[0]}')
print(f'Test size: {X_test.shape[0]}')

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Configure dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
val_acc = []
loss_epoch_C = []
top2_train_acc = []
top2_val_acc = []

# Early Stopping Parameters (optional)
patience = 200
best_epoch = 0

best_val_acc  = 0
best_model_state:dict = None

#%%
# Train the model
for epoch in (range(epochs)):
    correct_train, total_train = 0, 0
    
    top2_correct_train=0
    train_loss_C = 0

    for data, target in tqdm(train_dataloader):
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
        top2_correct_train += torch.sum(
            (predicted_top2[:, 0] == target) | (predicted_top2[:, 1] == target)
        ).item()

        train_loss_C += loss.item()

    # Calculate training accuracy
    train_accuracy = correct_train / total_train
    top2_train_accuracy = top2_correct_train / total_train
    train_acc.append(train_accuracy * 100)
    top2_train_acc.append(top2_train_accuracy * 100)
    loss_epoch_C.append(train_loss_C / len(train_dataloader))

    print(f'\nEpoch {epoch+1}/{epochs} - Training Loss: {train_loss_C/len(train_dataloader):.4f} | '
          f'Training Acc: {train_accuracy:.4f} | Top2 Acc: {top2_train_accuracy:.4f}')

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    top2_correct_val = 0
    val_loss_C = 0

    with torch.no_grad():
        correct = 0
        # total = 0
        for data, target in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
            
            # if torch.is_tensor(data) and data.dtype == torch.double:
            #     data = data.float()
                # target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            # print(output.data.shape)
            # total_test += target.size(0)
            # Calculate predictions
            _, predicted = torch.max(output.data, 1)
            _, predicted_top2 = torch.topk(output.data, 2, dim=1)
            
            # Update validation metrics
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()
            top2_correct_val += torch.sum(
                (predicted_top2[:, 0] == target) | (predicted_top2[:, 1] == target)
            ).item()

            val_loss_C += loss.item()
    # Calculate validation accuracy
    val_accuracy = correct_val / total_val
    top2_val_accuracy = top2_correct_val / total_val
    val_acc.append(val_accuracy * 100)
    top2_val_acc.append(top2_val_accuracy * 100)

    print(f'Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss_C/len(val_dataloader):.4f} | '
          f'Validation Acc: {val_accuracy:.4f} | Top2 Acc: {top2_val_accuracy:.4f}')
    
    # Save the best model based on validation accuracy
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_state = model.state_dict()
        best_epoch = epoch + 1
        print(f'--> Best model updated at epoch {best_epoch} with Validation Acc: {best_val_acc:.4f}')
        # Reset patience counter if using early stopping
        patience_counter = 0
    else:
        # Increment patience counter if using early stopping
        if 'patience_counter' in locals():
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

current_datetime = datetime.now()
year = current_datetime.year
month = current_datetime.month
day = current_datetime.day
hour = current_datetime.hour
minute = current_datetime.minute

# Save the model (optional)
torch.save(best_model_state, f"{month}_{day}_{hour}_{minute}_best_model_acc_{best_val_acc}.pth")

# After training, evaluate on test set and calculate metrics
all_preds = []
all_targets = []
all_preds_proba = []
# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'\nLoaded best model from epoch {best_epoch} with Validation Acc: {best_val_acc:.4f}')
else:
    print('\nNo improvement during training; using the last epoch model.')
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predicted = torch.argmax(output.data, dim=1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(torch.argmax(target, dim=1).cpu().numpy())
        all_preds_proba.extend(output.data.cpu().numpy())

# Confusion Matrix
conf_matrix = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'cnn1d_{month}_{day}_{hour}_{minute}_Confusion_Matrix.png')

# Classification Report
print("Classification Report:\n", classification_report(all_targets, all_preds))


# ROC-AUC Score and ROC Curve
y_test_labels = np.array(all_targets)
y_score = np.array(all_preds_proba)

# Binarize the output labels for multi-class ROC
y_test_binarized = label_binarize(y_test_labels, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_test_binarized.shape[1]

# One-vs-Rest ROC curve
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'cnn1d_{month}_{day}_{hour}_{minute}_ROC_curve.png')
plt.show()


# Plot Training and Validation Loss
plt.figure()
plt.plot(range(1, len(loss_epoch_C)+1), loss_epoch_C, label='Training Loss')
plt.plot(range(1, len(val_acc)+1), [val_loss / len(val_dataloader) for val_loss in loss_epoch_C], label='Validation Loss')  # Adjust as needed
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(f'cnn1d_{month}_{day}_{hour}_{minute}_loss.png')
plt.show()

# Plot Training and Validation Accuracy
plt.figure()
plt.plot(range(1, len(train_acc)+1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc)+1), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'cnn1d_{month}_{day}_{hour}_{minute}_accuracy.png')
plt.show()
# Plot Top-2 Training and Validation Accuracy
plt.figure()
plt.plot(range(1, len(top2_train_acc)+1), top2_train_acc, label='Top-2 Training Accuracy')
plt.plot(range(1, len(top2_val_acc)+1), top2_val_acc, label='Top-2 Validation Accuracy')
plt.title('Top-2 Training and Validation Accuracy')
plt.ylabel('Top-2 Accuracy (%)')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'cnn1d_{month}_{day}_{hour}_{minute}_top2_accuracy.png')
plt.show()
plt.close()


cursor.close()
connction.close()