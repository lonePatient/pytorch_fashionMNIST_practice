# #encoding:utf-8
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self,num_classes):
        super(SimpleNet, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        # Fully Connected 1
        self.fc1 = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        # Resize
        out = out.view(out.size(0), -1)
        # Dropout
        out = self.dropout(out)
        # Fully connected 1
        out = self.fc1(out)
        return out