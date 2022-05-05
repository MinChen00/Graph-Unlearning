import logging

import config
from lib_gnn_model.graphsage.graphsage import SAGE
from lib_dataset.data_store import DataStore


class NodeEmbedding:
    def __init__(self, args, graph, data):
        super(NodeEmbedding, self)

        self.logger = logging.getLogger(__name__)
        self.args = args
        self.graph = graph
        self.data = data

        self.data_store = DataStore(self.args)

    def sage_encoder(self):
        if self.args['is_gen_embedding']:
            self.logger.info("generating node embeddings with GraphSage...")

            node_to_embedding = {}
            # run sage
            self.target_model = SAGE(self.data.num_features, len(self.data.y.unique()), self.data)

            # self.target_model.train_model(50)

            # load a pretrained GNN model for generating node embeddings
            target_model_name = '_'.join((self.args['target_model'], 'random_1',
                                          str(self.args['shard_size_delta']),
                                          str(self.args['ratio_deleted_edges']), '0_0_1'))
            target_model_file = config.MODEL_PATH + self.args['dataset_name'] + '/' + target_model_name
            self.target_model.load_model(target_model_file)

            logits = self.target_model.generate_embeddings().detach().cpu().numpy()
            for node in self.graph.nodes:
                node_to_embedding[node] = logits[node]

            self.data_store.save_embeddings(node_to_embedding)
        else:
            node_to_embedding = self.data_store.load_embeddings()

        return node_to_embedding
