import logging
import time

import numpy as np

from exp.exp import Exp
from lib_gnn_model.graphsage.graphsage import SAGE
from lib_gnn_model.gat.gat import GAT
from lib_gnn_model.gin.gin import GIN
from lib_gnn_model.gcn.gcn import GCN
from lib_gnn_model.mlp.mlp import MLP
from lib_gnn_model.node_classifier import NodeClassifier
from lib_aggregator.aggregator import Aggregator


class ExpUnlearning(Exp):
    def __init__(self, args):
        super(ExpUnlearning, self).__init__(args)

        self.logger = logging.getLogger('exp_unlearning')

        self.target_model_name = self.args['target_model']
        self.num_opt_samples = self.args['num_opt_samples']

        self.load_data()
        self.determine_target_model()

        run_f1 = np.empty((0))
        unlearning_time = np.empty((0))
        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)
            self.train_target_models(run)
            aggregate_f1_score = self.aggregate(run)
            node_unlearning_time = self.unlearning_time_statistic()
            run_f1 = np.append(run_f1, aggregate_f1_score)
            unlearning_time = np.append(unlearning_time, node_unlearning_time)

        self.f1_score_avg = np.average(run_f1)
        self.f1_score_std = np.std(run_f1)
        self.unlearning_time_avg = np.average(unlearning_time)
        self.unlearning_time_std = np.std(unlearning_time)
        self.logger.info("%s %s %s %s" % (self.f1_score_avg, self.f1_score_std, self.unlearning_time_avg, self.unlearning_time_std))

    def load_data(self):
        self.shard_data = self.data_store.load_shard_data()
        self.data = self.data_store.load_raw_data()

    def determine_target_model(self):
        num_feats = self.data.num_features
        num_classes = len(self.data.y.unique())

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
            for shard in range(self.num_shards):
                self.time[shard] = self._train_model(run, shard)

    def aggregate(self, run):
        self.logger.info('aggregating submodels')

        start_time = time.time()
        aggregator = Aggregator(run, self.target_model, self.data, self.shard_data, self.args)
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate()
        aggregate_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % aggregate_time)

        self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score

    def unlearning_time_statistic(self):
        if self.args['is_train_target_model'] and self.num_shards != 1:
            self.community_to_node = self.data_store.load_community_data()
            node_list = []
            for key, value in self.community_to_node.items():
                node_list.extend(value)

            # random sample 5% nodes, find their belonging communities
            sample_nodes = np.random.choice(node_list, int(0.05 * len(node_list)))
            belong_community = []
            for sample_node in range(len(sample_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(sample_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.num_shards):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))])

            return node_unlearning_time

        elif self.args['is_train_target_model'] and self.num_shards == 1:
            return self.time[0]

        else:
            return 0

    def _train_model(self, run, shard):
        self.logger.info('training target models, run %s, shard %s' % (run, shard))

        start_time = time.time()
        self.target_model.data = self.shard_data[shard]
        self.target_model.train_model()
        train_time = time.time() - start_time

        self.data_store.save_target_model(run, self.target_model, shard)
        self.logger.info("Model training time: %s" % (train_time))

        return train_time
