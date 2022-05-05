import numpy as np

from lib_graph_partition.partition import Partition


class PartitionRandom(Partition):
    def __init__(self, args, graph):
        super(PartitionRandom, self).__init__(args, graph)

    def partition(self):
        graph_nodes = np.array(self.graph.nodes)
        np.random.shuffle(graph_nodes)
        train_shard_indices = np.array_split(graph_nodes, self.args['num_shards'])

        return dict(zip(range(self.num_shards), train_shard_indices))
