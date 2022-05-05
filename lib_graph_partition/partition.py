import numpy as np


class Partition:
    def __init__(self, args, graph, dataset=None):
        self.args = args
        self.graph = graph
        self.dataset = dataset

        self.partition_method = self.args['partition_method']
        self.num_shards = self.args['num_shards']
        self.dataset_name = self.args['dataset_name']
        
    def idx2id(self, idx_dict, node_list):
        ret_dict = {}
        for com, idx in idx_dict.items():
            ret_dict[com] = node_list[list(idx)]
        
        return ret_dict
    
    def id2idx(self, id_dict, node_list):
        ret_dict = {}
        for com, id in id_dict.items():
            ret_dict[com] = np.searchsorted(node_list, id)
        
        return ret_dict
