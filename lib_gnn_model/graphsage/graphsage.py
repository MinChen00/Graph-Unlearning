import os
import logging

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler

from lib_gnn_model.graphsage.graphsage_net import SageNet
from lib_gnn_model.gnn_base import GNNBase
import config


class SAGE(GNNBase):
    def __init__(self, num_feats, num_classes, data=None):
        super(SAGE, self).__init__()
        self.logger = logging.getLogger('graphsage')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = SageNet(num_feats, 256, num_classes).to(self.device)
        self.data = data

    def train_model(self, num_epochs=100):
        self.model.train()
        self.model.reset_parameters()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)
        self._gen_train_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.001)

        for epoch in range(num_epochs):
            self.logger.info('epoch %s' % (epoch,))

            for batch_size, n_id, adjs in self.train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(self.device) for adj in adjs]

                optimizer.zero_grad()
                out = self.model(self.data.x[n_id], adjs)
                loss = F.nll_loss(out, self.data.y[n_id[:batch_size]])
                loss.backward()
                optimizer.step()

            train_acc, test_acc = self.evaluate_model()
            self.logger.info(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader()

        out = self.model.inference(self.data.x, self.subgraph_loader, self.device)
        y_true = self.data.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        results = []
        for mask in [self.data.train_mask, self.data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

        return results

    def posterior(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader()

        posteriors = self.model.inference(self.data.x, self.subgraph_loader, self.device)

        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask], dim=-1)

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader()

        logits = self.model.inference(self.data.x, self.subgraph_loader, self.device)
        return logits

    def _gen_train_loader(self):
        if self.data.edge_index.shape[1] == 0:
            self.data.edge_index = torch.tensor([[1, 2], [2, 1]])
        self.train_loader = NeighborSampler(self.data.edge_index, node_idx=self.data.train_mask,
                                            # sizes=[25, 10], batch_size=128, shuffle=True,
                                            # sizes=[25, 10], num_nodes=self.data.num_nodes,
                                            sizes=[10, 10], num_nodes=self.data.num_nodes,
                                            # sizes=[5, 5], num_nodes=self.data.num_nodes,
                                            # batch_size=128, shuffle=True,
                                            batch_size=64, shuffle=True,
                                            num_workers=0)

    def _gen_subgraph_loader(self):
        self.subgraph_loader = NeighborSampler(self.data.edge_index, node_idx=None,
                                               # sizes=[-1], num_nodes=self.data.num_nodes,
                                               sizes=[10], num_nodes=self.data.num_nodes,
                                               # batch_size=128, shuffle=False,
                                               batch_size=64, shuffle=False,
                                               num_workers=0)


if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    graphsage = SAGE(dataset.num_features, dataset.num_classes, data)
    graphsage.train_model()
