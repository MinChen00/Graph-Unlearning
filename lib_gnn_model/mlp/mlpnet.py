from torch import nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPNet, self).__init__()
        self.xent = nn.CrossEntropyLoss()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.Linear(250, 100),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return F.softmax(x, dim=1)

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

    def reset_parameters(self):
        return 0
