import torch.nn as nn
import torch.nn.functional as F


def _glorot_init(model):
    """Normalized (Glorot/Xavier) initialization, as used in Reddi et al. 2016."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def MNIST_one_layer():
    # Create the nn model
    input_size = 784
    hidden_sizes = [100]
    output_size = 10

    # The non-convex function (model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], output_size))
    return _glorot_init(model)


def MNIST_two_layers():
    # Create the nn model
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # The non-convex function (model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size))
    return _glorot_init(model)


class MNIST_ConvNet(nn.Module):
    def __init__(self):
        super(MNIST_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 filters
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)         # Output layer (10 classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)  # Downsample
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)  # Downsample
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x