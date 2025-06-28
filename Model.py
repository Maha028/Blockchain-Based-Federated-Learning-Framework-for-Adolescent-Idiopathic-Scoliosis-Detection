import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoliosisCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ScoliosisCNN, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after conv2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization after conv3
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming input images are 224x224
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer with configurable rate
        self.fc2 = nn.Linear(512, 2)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BatchNorm -> ReLU -> Pooling
        
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor

        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.dropout(x)  # Apply Dropout
        x = self.fc2(x)  # Fully connected layer 2 (output)

        return x
