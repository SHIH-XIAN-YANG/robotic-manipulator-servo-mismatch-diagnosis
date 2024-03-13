import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

import numpy as np

class MobileNetV2(nn.Module):
    def __init__(self, in_channels=6, num_classes=6):
        super(MobileNetV2, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Inverted residual blocks
        self.block1 = self._make_block(32, 16, stride=1)
        self.block2 = self._make_block(16, 24, stride=2)
        self.block3 = self._make_block(24, 48, stride=2)
        self.block4 = self._make_block(48, 64, stride=2)
        self.block5 = self._make_block(64, 96, stride=1)
        self.block6 = self._make_block(96, 160, stride=2)
        self.block7 = self._make_block(160, 320, stride=2)

        # Final classification layer
        self.conv2 = nn.Conv2d(320, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_block(self, in_filters, out_filters, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x
    

# Define the model architecture
class CustomResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomResNet, self).__init__()
        self.base_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.base_model.conv1 = nn.Conv2d(3 * 2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Replace first conv with 3*2 channels
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        x = self.base_model(x)
        return x
    
class UltraLightCNN(nn.Module):
    def __init__(self, input_shape=6, num_classes=6):
        super(UltraLightCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNN1D(nn.Module):
    def __init__(self, input_dim=2, output_dim=6):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.batch_norm6 = nn.BatchNorm1d(512)
        self.batch_norm7 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.1)
        self.dropout6 = nn.Dropout(p=0.1)
        self.dropout7 = nn.Dropout(p=0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(83968, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(84544, output_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout3(x)
        # x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))
        # x = self.dropout4(x)
        # x = self.pool(self.relu(self.batch_norm5(self.conv5(x))))
        # x = self.dropout5(x)
        # x = self.pool(self.relu(self.batch_norm6(self.conv6(x))))
        # x = self.dropout6(x)
        # x = self.pool(self.relu(self.batch_norm7(self.conv7(x))))
        # x = self.dropout7(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__=="__main__":
    # Instantiate the model
    model = CNN1D(input_dim=6, output_dim=6)

    # Testing the model with random input
    random_input = torch.randn(64, 6, 10568)  # Assuming input size is 224x224
    output = model(random_input)
    print(output.shape)

