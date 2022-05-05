import logging
import time

import torch
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import Data
import torch_geometric as tg
import networkx as nx

from exp.exp import Exp
from lib_utils.utils import connected_component_subgraphs
from lib_graph_partition.graph_partition import GraphPartition
from lib_utils import utils


class ExpGraphPartition(Exp):
    def __init__(self, args):
        super(ExpGraphPartition, self).__init__(args)

        self.logger = logging.getLogger('exp_graph_partition')

        self.load_data()
        self.train_test_split()
        self.gen_train_graph()
        self.graph_partition()
        self.generate_shard_data()

    def load_data(self):
        self.data = self.data_store.load_raw_data()

    def train_test_split(self):
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
            self.data_store.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def gen_train_graph(self):
        # delete ratio of edges and update the train graph
        if self.args['ratio_deleted_edges'] != 0:
            self.logger.debug("Before edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data.num_nodes, self.data.num_edges))

            # self._ratio_delete_edges()
            self.data.edge_index = self._ratio_delete_edges(self.data.edge_index)

        # decouple train test edges.
        edge_index = self.data.edge_index.numpy()
        test_edge_indices = np.logical_or(np.isin(edge_index[0], self.test_indices),
                                          np.isin(edge_index[1], self.test_indices))
        train_edge_indices = np.logical_not(test_edge_indices)
        edge_index_train = edge_index[:, train_edge_indices]

        self.train_graph = nx.Graph()
        self.train_graph.add_nodes_from(self.train_indices)

        # use largest connected graph as train graph
        if self.args['is_prune']:
            self._prune_train_set()

        # reconstruct a networkx train graph
        for u, v in np.transpose(edge_index_train):
            self.train_graph.add_edge(u, v)

        self.logger.debug("After edge deletion. train graph  #.Nodes: %f, #.Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        self.logger.debug("After edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))
        self.data_store.save_train_data(self.data)
        self.data_store.save_train_graph(self.train_graph)

    def graph_partition(self):
        if self.args['is_partition']:
            self.logger.info('graph partitioning')

            start_time = time.time()
            partition = GraphPartition(self.args, self.train_graph, self.data)
            self.community_to_node = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.data_store.save_community_data(self.community_to_node)
        else:
            self.community_to_node = self.data_store.load_community_data()

    def generate_shard_data(self):
        self.logger.info('generating shard data')

        self.shard_data = {}
        for shard in range(self.args['num_shards']):
            train_shard_indices = list(self.community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = self.data.x[shard_indices]
            y = self.data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))

            self.shard_data[shard] = data

        self.data_store.save_shard_data(self.shard_data)

    def _prune_train_set(self):
        # extract the the maximum connected component
        self.logger.debug("Before Prune...  #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

        self.train_graph = max(connected_component_subgraphs(self.train_graph), key=len)

        self.logger.debug("After Prune... #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        # self.train_indices = np.array(self.train_graph.nodes)

    def _ratio_delete_edges(self, edge_index):
        edge_index = edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]
        remain_indices = np.random.choice(unique_indices,
                                           int(unique_indices.shape[0] * (1.0 - self.args['ratio_deleted_edges'])),
                                           replace=False)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        # self.data.edge_index = torch.from_numpy(edge_index[:, remain_indices])
        return torch.from_numpy(edge_index[:, remain_indices])
