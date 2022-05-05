import logging

from lib_graph_partition.partition_kmeans import PartitionKMeans
from lib_graph_partition.partition_lpa import PartitionConstrainedLPA, PartitionLPA, PartitionConstrainedLPABase
from lib_graph_partition.metis_partition import MetisPartition
from lib_graph_partition.partition_random import PartitionRandom


class GraphPartition:
    def __init__(self, args, graph, dataset=None):
        self.logger = logging.getLogger(__name__)

        self.args = args
        self.graph = graph
        self.dataset = dataset

        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']

    def graph_partition(self):
        self.logger.info('graph partition, method: %s' % self.partition_method)

        if self.partition_method == 'random':
            partition_method = PartitionRandom(self.args, self.graph)
        elif self.partition_method in ['sage_km', 'sage_km_base']:
            partition_method = PartitionKMeans(self.args, self.graph, self.dataset)
        elif self.partition_method == 'lpa' and not self.args['is_constrained']:
            partition_method = PartitionLPA(self.args, self.graph)
        elif self.partition_method == 'lpa' and self.args['is_constrained']:
            partition_method = PartitionConstrainedLPA(self.args, self.graph)
        elif self.partition_method == 'lpa_base':
            partition_method = PartitionConstrainedLPABase(self.args, self.graph)
        elif self.partition_method == 'metis':
            partition_method = MetisPartition(self.args, self.graph, self.dataset)
        else:
            raise Exception('Unsupported partition method')

        return partition_method.partition()
