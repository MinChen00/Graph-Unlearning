import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.data import Data
from scipy.spatial import distance

import config
from exp.exp import Exp
from lib_graph_partition.graph_partition import GraphPartition
from lib_gnn_model.node_classifier import NodeClassifier
from lib_aggregator.aggregator import Aggregator
from lib_utils import utils


class ExpAttackUnlearning(Exp):
    def __init__(self, args):
        super(ExpAttackUnlearning, self).__init__(args)
        self.logger = logging.getLogger('exp_attack_unlearning')
        # 1. respond to the unlearning requests
        self.load_preprocessed_data()
        # self.graph_unlearning_request_respond()
        if self.args['repartition']:
            with open(config.MODEL_PATH + self.args['dataset_name'] + '/' + self.args['target_model']+"_unlearned_indices") as file:
                node_unlearning_indices = [line.rstrip() for line in file]
            for unlearned_node in node_unlearning_indices:
                self.graph_unlearning_request_respond(int(unlearned_node))
        else:
            self.graph_unlearning_request_respond()
        # 2. evalute the attack performance
        self.attack_graph_unlearning()

    def load_preprocessed_data(self):
        self.shard_data = self.data_store.load_shard_data()
        self.raw_data = self.data_store.load_raw_data()
        self.train_data = self.data_store.load_train_data()
        self.train_graph = self.data_store.load_train_graph()
        self.train_indices, self.test_indices = self.data_store.load_train_test_split()
        self.community_to_node = self.data_store.load_community_data()
        num_feats = self.train_data.num_features
        num_classes = len(self.train_data.y.unique())
        self.target_model = NodeClassifier(num_feats, num_classes, self.args)

    def graph_unlearning_request_respond(self, node_unlearning_request=None):
        # reindex the node ids
        node_to_com = self.data_store.c2n_to_n2c(self.community_to_node)
        train_indices_prune = list(node_to_com.keys())

        if node_unlearning_request==None:
            # generate node unlearning requests       
            node_unlearning_indices = np.random.choice(train_indices_prune, self.args['num_unlearned_nodes'])
        else:
            node_unlearning_indices = np.array([node_unlearning_request])
        self.num_unlearned_edges =0
        unlearning_indices = defaultdict(list)
        for node in node_unlearning_indices:
                unlearning_indices[node_to_com[node]].append(node)
        # delete a list of revoked nodes from train_graph
        self.train_graph.remove_nodes_from(node_unlearning_indices)        

        # delete the revoked nodes from train_data 
        # by building unlearned data from unlearned train_graph
        self.train_data.train_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), self.train_indices))
        self.train_data.test_mask = torch.from_numpy(np.isin(np.arange(self.train_data.num_nodes), np.append(self.test_indices, node_unlearning_indices)))

        # delete the revoked nodes from shard_data
        self.shard_data_after_unlearning = {}
        self.affected_shard=[]
        for shard in range(self.args["num_shards"]):
            train_shard_indices = list(self.community_to_node[shard])
            # node unlearning
            train_shard_indices = np.setdiff1d(train_shard_indices, unlearning_indices[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.train_data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))

            self.shard_data_after_unlearning[shard] = data
            self.num_unlearned_edges += self.shard_data[shard].num_edges - self.shard_data_after_unlearning[shard].num_edges

            # find the affected shard model      
            if self.shard_data_after_unlearning[shard].num_nodes != self.shard_data[shard].num_nodes:
                self.affected_shard.append(shard)
        
        self.data_store.save_unlearned_data(self.train_graph, 'train_graph')
        self.data_store.save_unlearned_data(self.train_data, 'train_data')
        self.data_store.save_unlearned_data(self.shard_data_after_unlearning, 'shard_data')

        # retrain the correponding shard model
        if not self.args['repartition']:
            for shard in self.affected_shard:
                suffix = "unlearned_"+str(node_unlearning_indices[0])
                self._train_shard_model(shard, suffix)

        # (if re-partition, re-partition the remaining graph)
        # re-train the shard model, save model and optimal weight score
        if self.args['repartition']:
            suffix="_repartition_unlearned_" + str(node_unlearning_indices[0])
            self._repartition(suffix)
            for shard in range(self.args["num_shards"]):
                self._train_shard_model(shard, suffix)

    def _repartition(self, suffix):
        # load unlearned train_graph and train_data
        train_graph = self.data_store.load_unlearned_data('train_graph')
        train_data = self.data_store.load_unlearned_data('train_data')
        # repartition
        start_time = time.time()
        partition = GraphPartition(self.args, train_graph, train_data)
        community_to_node = partition.graph_partition()
        partition_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % partition_time)
        # save the new partition and shard
        self.data_store.save_community_data(community_to_node, suffix)       
        self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, self.test_indices)

    def _generate_unlearned_repartitioned_shard_data(self, train_data, community_to_node, test_indices):
        self.logger.info('generating shard data')

        shard_data = {}
        for shard in range(self.args['num_shards']):
            train_shard_indices = list(community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, test_indices)

            x = self.train_data.x[shard_indices]
            y = self.train_data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(train_data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, test_indices))

            shard_data[shard] = data

        # self.data_store.save_unlearned_data(shard_data, 'shard_data_repartition')
        return shard_data
    
    def _train_shard_model(self, shard, suffix="unlearned"):
        self.logger.info('training target models, shard %s' % shard)

        # load shard data
        self.target_model.data = self.shard_data_after_unlearning[shard]
        # retrain shard model
        self.target_model.train_model()
        # replace shard model
        device=torch.device("cpu")
        self.target_model.device = device
        self.data_store.save_target_model(0, self.target_model, shard, suffix)
        # self.data_store.save_unlearned_target_model(0, self.target_model, shard, suffix)

    def attack_graph_unlearning(self):

        # load unlearned indices
        with open(config.MODEL_PATH + self.args['dataset_name'] + "/" + self.args['target_model'] +"_unlearned_indices") as file:
            unlearned_indices = [line.rstrip() for line in file]

        # member sample query, label as 1
        positive_posteriors = self._query_target_model(unlearned_indices, unlearned_indices)
        # non-member sample query, label as 0
        negative_posteriors = self._query_target_model(unlearned_indices, self.test_indices)

        # evaluate attack performance, train multiple shadow models, or calculate posterior entropy, or directly calculate AUC.
        self.evaluate_attack_performance(positive_posteriors, negative_posteriors)

    def _query_target_model(self, unlearned_indices, test_indices):
        # load unlearned data
        train_data = self.data_store.load_unlearned_data('train_data')

        # load optimal weight score
        # optimal_weight=self.data_store.load_optimal_weight(0)

        # calculate the final posterior, save as attack feature
        self.logger.info('aggregating submodels')
        posteriors_a, posteriors_b, posteriors_c =[],[],[]

        for i in unlearned_indices:
            community_to_node = self.data_store.load_community_data('')
            shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, int(i))        

            posteriors_a.append(self._generate_posteriors(shard_data, ''))            
 
            suffix="unlearned_" + str(i)
            posteriors_b.append(self._generate_posteriors_unlearned(shard_data, suffix, i))

            if self.args['repartition']:
                suffix = "_repartition_unlearned_" + str(i)
                community_to_node = self.data_store.load_community_data(suffix)
                shard_data = self._generate_unlearned_repartitioned_shard_data(train_data, community_to_node, int(i))        
                suffix = "__repartition_unlearned_" + str(i)
                posteriors_c.append(self._generate_posteriors(shard_data, suffix))

        return posteriors_a, posteriors_b, posteriors_c

    def _generate_posteriors_unlearned(self, shard_data, suffix, unlearned_indice):
        import glob
        model_path=glob.glob(config.MODEL_PATH+self.args['dataset_name']+"/*_1unlearned_"+str(unlearned_indice))
        if not model_path:
            self.logger.info("No corresponding unlearned shard model for node %s" % str(unlearned_indice))
            return torch.tensor([0]*6)
        else:
            affected_shard = int(model_path[0].split('/')[-1].split('_')[-4])
            posteriors = []
            for shard in range(self.args['num_shards']):
                if shard == affected_shard:
                    # load the retrained the shard model
                    self.data_store.load_target_model(0, self.target_model, shard, suffix)
                else:
                    # self.target_model.model.reset_parameters()
                    # load unaffected shard model
                    self.data_store.load_target_model(0, self.target_model, shard, '')
                self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
                self.target_model.model = self.target_model.model.to(self.device)         
                self.target_model.data = shard_data[shard].to(self.device)
                posteriors.append(self.target_model.posterior())
            return torch.mean(torch.cat(posteriors, dim=0), dim=0)

    def _generate_posteriors(self, shard_data, suffix):
        posteriors = []
        for shard in range(self.args['num_shards']):
            # self.target_model.model.reset_parameters()
            self.data_store.load_target_model(0, self.target_model, shard, suffix)
            self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
            self.target_model.model = self.target_model.model.to(self.device)         
            self.target_model.data = shard_data[shard].to(self.device)

            posteriors.append(self.target_model.posterior())
        return torch.mean(torch.cat(posteriors, dim=0), dim=0)

    def evaluate_attack_performance(self, positive_posteriors, negative_posteriors):
        # constrcut attack data
        label = torch.cat((torch.ones(len(positive_posteriors[0])), torch.zeros(len(negative_posteriors[0]))))
        data={}
        for i in range(2):
            data[i] = torch.cat((torch.stack(positive_posteriors[i]), torch.stack(negative_posteriors[i])),0)

        # calculate l2 distance
        model_b_distance = self._calculate_distance(data[0], data[1])
        # directly calculate AUC with feature and labels
        attack_auc_b = self.evaluate_attack_with_AUC(model_b_distance, label)

        if self.args['repartition']:
            model_c_distance = self._calculate_distance(data[0], data[2])
            attack_auc_c = self.evaluate_attack_with_AUC(model_c_distance, label)

        self.logger.info("Attack_Model_B AUC: %s | Attack_Model_C AUC: %s" % (attack_auc_b, attack_auc_c))

    def evaluate_attack_with_AUC(self, data, label):
        from sklearn.metrics import roc_auc_score
        self.logger.info("Directly calculate the attack AUC")
        return roc_auc_score(label, data.reshape(-1, 1))
    
    def _calculate_distance(self, data0, data1, distance='l2_norm' ):
        if distance == 'l2_norm':
            return np.array([np.linalg.norm(data0[i]-data1[i]) for i in range(len(data0))])
        elif distance =='direct_diff':
            return data0 - data1
        else:
            raise Exception("Unsupported distance")
