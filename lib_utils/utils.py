import os
import errno

import numpy as np
import pandas as pd
import networkx as nx
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm


def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    return graph


def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index) + 1
    feature_count = max(feature_index) + 1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    return features


def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
    return target


def make_adjacency(graph, max_degree, sel=None):
    all_nodes = np.array(graph.nodes())

    # Initialize w/ links to a dummy node
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)

    if sel is not None:
        # only look at nodes in training set
        all_nodes = all_nodes[sel]

    for node in tqdm(all_nodes):
        neibs = np.array(list(graph.neighbors(node)))

        if sel is not None:
            neibs = neibs[sel[neibs]]

        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                extra = np.random.choice(neibs, max_degree - neibs.shape[0], replace=True)
                neibs = np.concatenate([neibs, extra])
            adj[node, :] = neibs

    return adj


def connected_component_subgraphs(graph):
    """
    Find all connected subgraphs in a networkx Graph

    Args:
        graph (Graph): A networkx Graph

    Yields:
        generator: A subgraph generator
    """
    for c in nx.connected_components(graph):
        yield graph.subgraph(c)


def check_exist(file_name):
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index


def pyg_to_nx(data):
    """
    Convert a torch geometric Data to networkx Graph.

    Args:
        data (Data): A torch geometric Data.

    Returns:
        Graph: A networkx Graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(data.num_nodes))
    edge_index = data.edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def edge_index_to_nx(edge_index, num_nodes):
    """
    Convert a torch geometric Data to networkx Graph by edge_index.
    Args:
        edge_index (Data.edge_index): A torch geometric Data.
        num_nodes (int): Number of nodes in a graph.
    Returns:
        Graph: networkx Graph
    """
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(num_nodes))
    edge_index = edge_index.numpy()

    for u, v in np.transpose(edge_index):
        graph.add_edge(u, v)

    return graph


def filter_edge_index_1(data, node_indices):
    """
    Remove unnecessary edges from a torch geometric Data, only keep the edges between node_indices.
    Args:
        data (Data): A torch geometric Data.
        node_indices (list): A list of nodes to be deleted from data.

    Returns:
        data.edge_index: The new edge_index after removing the node_indices.
    """
    if isinstance(data.edge_index, torch.Tensor):
        data.edge_index = data.edge_index.cpu()

    edge_index = data.edge_index
    node_index = np.isin(edge_index, node_indices)

    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = data.edge_index[:, col_index]

    return np.searchsorted(node_indices, edge_index)
