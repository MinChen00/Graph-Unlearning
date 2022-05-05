import os
import logging

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import RandomNodeSampler
from torch_geometric.datasets import Planetoid

from lib_gnn_model.gnn_base import GNNBase
from lib_gnn_model.gcn.gcn_net import GCNNet
import config


class GCN(GNNBase):
    def __init__(self, num_feats, num_classes, data=None):
        super(GCN, self).__init__()
        self.logger = logging.getLogger('gcn')

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.model = GCNNet(num_feats, num_classes)
        self.data = data

    def train_model(self, num_epoch=100):
        self.model.train()
        self.model.reset_parameters()
        self.model = self.model.to(self.device)
        self._gen_train_loader()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(num_epoch):
            self.logger.info('epoch %s' % (epoch,))

            for data in self.train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            train_acc, test_acc = self.evaluate_model()
            self.logger.info('train acc: %s, test acc: %s' % (train_acc, test_acc))

    def evaluate_model(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self._gen_test_loader()

        y_true = {'train': [], 'test': []}
        y_pred = {'train': [], 'test': []}

        for data in self.test_loader:
            data = data.to(self.device)
            out = self.model(data)

            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())

        train_acc = torch.cat(y_pred['train'], dim=0).eq(torch.cat(y_true['train'], dim=0)).sum().item() / len(y_pred['train'])
        test_acc = torch.cat(y_pred['test'], dim=0).eq(torch.cat(y_true['test'], dim=0)).sum().item() / len(y_pred['test'])

        return train_acc, test_acc

    def _gen_train_loader(self):
        self.train_loader = RandomNodeSampler(self.data, num_parts=3, shuffle=True, num_workers=0)

    def _gen_test_loader(self):
        self.test_loader = RandomNodeSampler(self.data, num_parts=3, num_workers=0)


if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    gcn = GCN(dataset.num_features, dataset.num_classes, data)
    gcn.train_model()
