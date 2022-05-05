import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GCNNet, self).__init__()

        self.conv1 = GCNConv(num_feats, 16, cached=True, add_self_loops=False)
        self.conv2 = GCNConv(16, num_classes, cached=True, add_self_loops=False)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
