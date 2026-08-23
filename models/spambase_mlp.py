import torch


class SpambaseMLP(torch.nn.Module):
    def __init__(self, input_dim=57, hidden_dim=32, output_dim=2):
        super(SpambaseMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def get_spambase_mlp(input_dim=57, hidden_dim=32, output_dim=2):
    return SpambaseMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
