import numpy as np
import networkx as nx
import pymetis
from torch_geometric.data import ClusterData
from torch_geometric.utils import from_networkx

from lib_graph_partition.partition import Partition


class MetisPartition(Partition):
    def __init__(self, args, graph, dataset):
        super(MetisPartition, self).__init__(args, graph, dataset)
        self.graph = graph
        self.args = args
        self.data = dataset

    def partition(self, recursive=False):
        # recursive (bool, optional): If set to :obj:`True`, will use multilevel
        # recursive bisection instead of multilevel k-way partitioning.
        # (default: :obj:`False`)
        # only use train data, not the whole dataset
        self.train_data = from_networkx(self.graph)
        data = ClusterData(self.train_data, self.args['num_shards'], recursive=recursive)

        community_to_node = {}
        for i in range(self.args['num_shards']):
            community_to_node[i] = [*range(data.partptr[i], data.partptr[i+1], 1)]

        # map node back to original graph
        for com in range(self.args['num_shards']):
            community_to_node[com] = np.array(list(self.graph.nodes))[data.partptr.numpy()[com]:data.partptr.numpy()[com+1]]

        return community_to_node


class PyMetisPartition(Partition):
    def __init__(self, args, graph, dataset):
        super(PyMetisPartition, self).__init__(args, graph, dataset)
        self.graph = graph
        self.args = args
        self.data = dataset

    def partition(self, recursive=False):
        # recursive (bool, optional): If set to :obj:`True`, will use multilevel
        # recursive bisection instead of multilevel k-way partitioning.
        # (default: :obj:`False`)
        # only use train data, not the whole dataset
        # map graph into new graph
        mapping = {}
        for i, node in enumerate(self.graph.nodes):
            mapping[node] = i
        partition_graph = nx.relabel_nodes(self.graph, mapping=mapping)

        adj_list = []
        for line in nx.generate_adjlist(partition_graph):
            line_int = list(map(int, line.split()))
            adj_list.append(np.array(line_int))

        n_cuts, membership = pymetis.part_graph(self.args['num_shards'], adjacency=adj_list)

        # map node back to original graph
        community_to_node = {}
        for shard_index in range(self.args['num_shards']):
            community_to_node[shard_index] = np.array([node_id for node_id, node_shard_index in zip(list(mapping.keys()), membership) if node_shard_index == shard_index])
        return community_to_node
