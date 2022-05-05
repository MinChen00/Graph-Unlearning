import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv


class GINNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GINNet, self).__init__()

        dim = 32
        self.num_layers = 2

        nn1 = Sequential(Linear(num_feats, dim), ReLU(), Linear(dim, dim))
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn1))
        self.convs.append(GINConv(nn2))

        self.bn = torch.nn.ModuleList()
        self.bn.append(torch.nn.BatchNorm1d(dim))
        self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

            x = self.bn[i](x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

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

                x = self.bn[i](x)

                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        x_all = F.relu(self.fc1(x_all))
        x_all = self.fc2(x_all)

        return x_all.cpu()

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
