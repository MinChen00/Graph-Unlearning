import logging
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import config
from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.gat.gat_net import GATNet


class GAT(GNNBase):
    def __init__(self, num_feats, num_classes, data=None):
        super(GAT, self).__init__()
        self.logger = logging.getLogger('gat')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GATNet(num_feats, num_classes)
        self.data = data

    def train_model(self, num_epoch=100):
        self.model.train()
        self.model.reset_parameters()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=0.0001)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))

            optimizer.zero_grad()
            output = self.model(self.data)[self.data.train_mask]
            loss = F.nll_loss(output, self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()

            train_acc, test_acc = self.evaluate_model()
            self.logger.info('train acc: %s, test acc: %s' % (train_acc, test_acc))

    def evaluate_model(self):
        self.model.eval()
        # self.model, self.data = self.model.to(self.device), self.data.to(self.device)

        logits, accs = self.model(self.data), []

        for _, mask in self.data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs


if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    gat = GAT(dataset.num_features, dataset.num_classes, data)
    gat.train_model()
    # gat.evaluate_model()
