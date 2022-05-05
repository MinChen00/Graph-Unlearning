import logging
import pickle

import torch


class GNNBase:
    def __init__(self):
        self.logger = logging.getLogger('gnn')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model = None
        self.embedding_dim = 0
        self.data = None
        self.subgraph_loader = None

    def save_model(self, save_path):
        self.logger.info('saving model')
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        self.logger.info('loading model')
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(save_path, map_location=device))

    def save_paras(self, save_path):
        self.logger.info('saving paras')
        self.paras = {
            'embedding_dim': self.embedding_dim
        }
        pickle.dump(self.paras, open(save_path, 'wb'))

    def load_paras(self, save_path):
        self.logger.info('loading paras')
        return pickle.load(open(save_path, 'rb'))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def posterior(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        posteriors = self.model(self.data)
        for _, mask in self.data('test_mask'):
            posteriors = posteriors[mask]

        return posteriors.detach()
