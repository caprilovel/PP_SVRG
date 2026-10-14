import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .mnist_models import _glorot_init


def CIFAR10_one_layer():
    """One fully-connected hidden layer of 100 nodes, as in Reddi et al. 2016
    (the paper uses this same architecture for CIFAR-10, MNIST, and STL-10 --
    not a CNN)."""
    input_size = 3 * 32 * 32
    hidden_sizes = [100]
    output_size = 10

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], output_size))
    return _glorot_init(model)


class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def get_cifar10_ResNet18():
    return models.resnet18(num_classes=10)

class CIFAR10_LeNet5(nn.Module):
    def __init__(self):
        super(CIFAR10_LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CIFAR10_WideResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   