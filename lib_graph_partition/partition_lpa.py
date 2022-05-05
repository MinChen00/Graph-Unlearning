import math
import numpy as np
import networkx as nx
import logging
import pickle

from lib_graph_partition.constrained_lpa_base import ConstrainedLPABase
from lib_graph_partition.partition import Partition
from lib_graph_partition.constrained_lpa import ConstrainedLPA
import config


class PartitionLPA(Partition):
    def __init__(self, args, graph):
        super(PartitionLPA, self).__init__(args, graph)

        self.logger = logging.getLogger('partition_lpa')

    def partition(self):
        # implement LPA by hand, refer to https://github.com/benedekrozemberczki/LabelPropagation
        community_generator = nx.algorithms.community.label_propagation.label_propagation_communities(self.graph)
        self.logger.info("Generating LPA communities.")
        community_to_node = {key: c for key, c in zip(range(self.graph.number_of_nodes()), community_generator)}
        print("Found %s communities by unconstrained LPA", len(community_to_node.keys()))
        return community_to_node


class PartitionConstrainedLPA(Partition):
    def __init__(self, args, graph):
        super(PartitionConstrainedLPA, self).__init__(args, graph)
        self.args = args

        self.logger = logging.getLogger('partition_constrained_lpa')

    def partition(self):
        adj_array = nx.linalg.adj_matrix(self.graph).toarray().astype(np.bool)
        # node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards']) + 0.05 * self.graph.number_of_nodes()
        # node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'])
        node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'] +
                                   self.args['shard_size_delta'] * (self.graph.number_of_nodes()-self.graph.number_of_nodes() / self.args['num_shards']))

        self.logger.info(" #. nodes: %s. LPA shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))
        lpa = ConstrainedLPA(adj_array, self.num_shards, node_threshold, self.args['terminate_delta'])

        lpa.initialization()
        community_to_node, lpa_deltas = lpa.community_detection()

        pickle.dump(lpa_deltas, open(config.ANALYSIS_PATH + "partition/blpa_" + self.args['dataset_name'], 'wb'))

        return self.idx2id(community_to_node, np.array(self.graph.nodes))


class PartitionConstrainedLPABase(Partition):
    def __init__(self, args, graph):
        super(PartitionConstrainedLPABase, self).__init__(args, graph)
        self.args = args

        self.logger = logging.getLogger('partition_constrained_lpa')

    def partition(self):
        adj_array = nx.linalg.adj_matrix(self.graph).toarray().astype(np.bool)
        node_threshold = math.ceil(self.graph.number_of_nodes() / self.args['num_shards'] + self.args['shard_size_delta'] * (self.graph.number_of_nodes()-self.graph.number_of_nodes() / self.args['num_shards']))

        self.logger.info(" #. nodes: %s. LPA shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))
        lpa = ConstrainedLPABase(adj_array, self.num_shards, node_threshold, self.args['terminate_delta'])

        lpa.initialization()
        community_to_node, lpa_deltas = lpa.community_detection()

        pickle.dump(lpa_deltas, open(config.ANALYSIS_PATH + "partition/base_blpa_" + self.args['dataset_name'], 'wb'))

        return self.idx2id(community_to_node, np.array(self.graph.nodes))
