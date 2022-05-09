import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = 'cuda:1'#'cpu'
device = torch.device(device)


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(2,2), stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(1,1), stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(448, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        self.activate = nn.Softmax(dim=1)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, label = None):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)


        # return out
        if label != None:
            _, label = label.max(-1)
            pred = self.activate(out)
            loss = self.criteria(out, label)
            return loss, out, pred, label
        else:
            pred = self.activate(out)
            _, pred_label = pred.max(-1)
            return pred_label