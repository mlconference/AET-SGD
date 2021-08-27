import torch.nn as nn
import torch.nn.functional as func
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class myLeNet(nn.Module):
    def __init__(self):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        
        # Add dropout function.
        # self.dropout1 = nn.Dropout2d(0.2)
        #~ Dropout

        # self.fc1 = nn.Linear(4*4*50, 500)
        # self.fc2 = nn.Linear(500, 10)

        self.fc1 = nn.Linear(4*4*50, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))

        # Dropout
        # x = self.dropout1(x)
        #~Dropout

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x
    
    def name(self):
        return "myLeNet"