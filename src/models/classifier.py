# models/classifier.py

import torch
import torch.nn as nn

class ImprovedClassifier(nn.Module):
    """Neural network classifier for geolocation prediction."""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.225):
        super(ImprovedClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.dropout1(nn.ReLU()(self.bn1(self.fc1(x))))
        x = self.dropout2(nn.ReLU()(self.bn2(self.fc2(x))))
        x = self.dropout3(nn.ReLU()(self.bn3(self.fc3(x))))
        return self.fc4(x)
