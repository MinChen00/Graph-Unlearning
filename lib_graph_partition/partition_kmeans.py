import math
import pickle

import cupy as cp
import numpy as np
import logging

from sklearn.cluster import KMeans

import config
from lib_graph_partition.constrained_kmeans_base import ConstrainedKmeansBase
from lib_graph_partition.partition import Partition
from lib_graph_partition.constrained_kmeans import ConstrainedKmeans
from lib_node_embedding.node_embedding import NodeEmbedding


class PartitionKMeans(Partition):
    def __init__(self, args, graph, dataset):
        super(PartitionKMeans, self).__init__(args, graph, dataset)

        self.logger = logging.getLogger('partition_kmeans')
        cp.cuda.Device(self.args['cuda']).use()
        self.load_embeddings()

    def load_embeddings(self):
        node_embedding = NodeEmbedding(self.args, self.graph, self.dataset)

        if self.partition_method in ["sage_km", "sage_km_base"]:
            self.node_to_embedding = node_embedding.sage_encoder()
        else:
            raise Exception('unsupported embedding method')

    def partition(self):
        self.logger.info("partitioning")

        embedding = []
        for node in self.node_to_embedding.keys():
            embedding.append(self.node_to_embedding[node])

        if not self.args['is_constrained']:
            cluster = KMeans(n_clusters=self.num_shards, random_state=10)
            cluster_labels = cluster.fit_predict(embedding)

            node_to_community = {}
            for com, node in zip(cluster_labels, self.node_to_embedding.keys()):
                node_to_community[node] = com

            community_to_node = {}
            for com in range(len(set(node_to_community.values()))):
                community_to_node[com] = np.where(np.array(list(node_to_community.values())) == com)[0]
            community_to_node = dict(sorted(community_to_node.items()))

        else:
            # node_threshold = math.ceil(self.graph.number_of_nodes() / self.num_shards)
            # node_threshold = math.ceil(self.graph.number_of_nodes() / self.num_shards + 0.05*self.graph.number_of_nodes())
            node_threshold = math.ceil(
                self.graph.number_of_nodes() / self.args['num_shards'] + self.args['shard_size_delta'] * (
                            self.graph.number_of_nodes() - self.graph.number_of_nodes() / self.args['num_shards']))
            self.logger.info("#.nodes: %s. Shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))

            if self.partition_method == 'sage_km_base':
                cluster = ConstrainedKmeansBase(np.array(embedding), num_clusters=self.num_shards,
                                                node_threshold=node_threshold,
                                                terminate_delta=self.args['terminate_delta'])
                cluster.initialization()
                community, km_deltas = cluster.clustering()
                pickle.dump(km_deltas, open(config.ANALYSIS_PATH + "partition/base_bkm_" + self.args['dataset_name'], 'wb'))

                community_to_node = {}
                for i in range(self.num_shards):
                    community_to_node[i] = np.array(community[i])

            if self.partition_method == 'sage_km':
                cluster = ConstrainedKmeans(cp.array(embedding), num_clusters=self.num_shards,
                                               node_threshold=node_threshold,
                                               terminate_delta=self.args['terminate_delta'])
                cluster.initialization()
                community, km_deltas = cluster.clustering()
                pickle.dump(km_deltas, open(config.ANALYSIS_PATH + "partition/bkm_" + self.args['dataset_name'], 'wb'))

                community_to_node = {}
                for i in range(self.num_shards):
                    community_to_node[i] = np.array(community[i].get().astype(int))

        return community_to_node

