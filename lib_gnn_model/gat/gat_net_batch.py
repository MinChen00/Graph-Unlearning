import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, dropout=0.6):
        super(GATNet, self).__init__()
        self.dropout = dropout

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_feats, 8, heads=8, dropout=self.dropout, add_self_loops=True))
        # On the Pubmed dataset, use heads=8 in conv2.
        self.convs.append(GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=self.dropout, add_self_loops=True))

    def forward(self, x, adjs):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
