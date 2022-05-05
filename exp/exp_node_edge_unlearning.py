import logging
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

import config
from exp.exp import Exp
from lib_gnn_model.graphsage.graphsage import SAGE
from lib_gnn_model.gat.gat import GAT
from lib_gnn_model.gin.gin import GIN
from lib_gnn_model.gcn.gcn import GCN
from lib_gnn_model.mlp.mlp import MLP
from lib_gnn_model.node_classifier import NodeClassifier
from lib_aggregator.aggregator import Aggregator
from lib_utils import utils


class ExpNodeEdgeUnlearning(Exp):
    def __init__(self, args):
        super(ExpNodeEdgeUnlearning, self).__init__(args)
        self.logger = logging.getLogger('exp_node_edge_unlearning')
        self.target_model_name = self.args['target_model']

        self.load_data()
        self.determine_target_model()
        self.run_exp()

    def run_exp(self):
        # unlearning efficiency
        run_f1 = np.empty((0))
        unlearning_time = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)
            self.train_target_models(run)
            aggregate_f1_score = self.aggregate(run)
            # node_unlearning_time = self.unlearning_time_statistic()
            node_unlearning_time = 0
            run_f1 = np.append(run_f1, aggregate_f1_score)
            unlearning_time = np.append(unlearning_time, node_unlearning_time)
        self.num_unlearned_edges = 0
        # model utility
        self.f1_score_avg = np.average(run_f1)
        self.f1_score_std = np.std(run_f1)
        self.unlearning_time_avg = np.average(unlearning_time)
        self.unlearning_time_std = np.std(unlearning_time)
        self.logger.info(
            "%s %s %s %s" % (self.f1_score_avg, self.f1_score_std, self.unlearning_time_avg, self.unlearning_time_std))

    def load_data(self):
        self.shard_data = self.data_store.load_shard_data()
        self.raw_data = self.data_store.load_raw_data()
        self.train_data = self.data_store.load_train_data()

        self.unlearned_shard_data = self.shard_data

    def determine_target_model(self):
        num_feats = self.train_data.num_features
        num_classes = len(self.train_data.y.unique())

        if not self.args['is_use_batch']:
            if self.target_model_name == 'SAGE':
                self.target_model = SAGE(num_feats, num_classes)
            elif self.target_model_name == 'GCN':
                self.target_model = GCN(num_feats, num_classes)
            elif self.target_model_name == 'GAT':
                self.target_model = GAT(num_feats, num_classes)
            elif self.target_model_name == 'GIN':
                self.target_model = GIN(num_feats, num_classes)
            else:
                raise Exception('unsupported target model')
        else:
            if self.target_model_name == 'MLP':
                self.target_model = MLP(num_feats, num_classes)
            else:
                self.target_model = NodeClassifier(num_feats, num_classes, self.args)

    def train_target_models(self, run):
        if self.args['is_train_target_model']:
            self.logger.info('training target models')

            self.time = {}
            for shard in range(self.args['num_shards']):
                self.time[shard] = self._train_model(run, shard)

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(run, self.target_model, self.train_data, self.unlearned_shard_data, self.args)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate()

        self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score

    def _generate_unlearning_request(self, num_unlearned="assign"):
        node_list = []
        for key, value in self.community_to_node.items():
            # node_list.extend(value.tolist())
            node_list.extend(value)
        if num_unlearned == "assign":
            num_of_unlearned_nodes = self.args['num_unlearned_nodes']
        elif num_unlearned == "ratio":
            num_of_unlearned_nodes = int(self.args['ratio_unlearned_nodes'] * len(node_list))

        if self.args['unlearning_request'] == 'random':
            unlearned_nodes_indices = np.random.choice(node_list, num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'top1':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True)
            unlearned_nodes_indices = np.random.choice(sorted_shards[0][1], num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'adaptive':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True)
            candidate_list = np.concatenate([sorted_shards[i][1] for i in range(int(self.args['num_shards']/2)+1)], axis=0)
            unlearned_nodes_indices = np.random.choice(candidate_list, num_of_unlearned_nodes, replace=False)

        elif self.args['unlearning_request'] == 'last5':
            sorted_shards = sorted(self.community_to_node.items(), key=lambda x: len(x[1]), reverse=False)
            candidate_list = np.concatenate([sorted_shards[i][1] for i in range(int(self.args['num_shards']/2)+1)], axis=0)
            unlearned_nodes_indices = np.random.choice(candidate_list, num_of_unlearned_nodes, replace=False)

        return unlearned_nodes_indices

    def unlearning_time_statistic(self):
        if self.args['is_train_target_model'] and self.args['num_shards'] != 1:
            # random sample 5% nodes, find their belonging communities
            unlearned_nodes = self._generate_unlearning_request(num_unlearned="ratio")
            belong_community = []
            for sample_node in range(len(unlearned_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(unlearned_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.args['num_shards']):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))])
            return node_unlearning_time

        elif self.args['is_train_target_model'] and self.args['num_shards'] == 1:
            return self.time[0]

        else:
            return 0

    def _train_model(self, run, shard):
        self.logger.info('training target models, run %s, shard %s' % (run, shard))

        start_time = time.time()
        self.target_model.data = self.unlearned_shard_data[shard]
        self.target_model.train_model()
        train_time = time.time() - start_time

        self.data_store.save_target_model(run, self.target_model, shard)

        return train_time
