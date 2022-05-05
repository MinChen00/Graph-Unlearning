import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, dropout=0.6):
        super(GATNet, self).__init__()
        self.dropout = dropout

        self.conv1 = GATConv(num_feats, 8, heads=8, dropout=self.dropout, add_self_loops=False)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=self.dropout, add_self_loops=False)
        # self.conv2 = GATConv(8 * 8, num_classes, heads=8, concat=False, dropout=self.dropout, add_self_loops=False)

        self.reset_parameters()

    def forward(self, data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
