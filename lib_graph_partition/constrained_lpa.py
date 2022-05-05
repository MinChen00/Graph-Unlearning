import copy
import logging
from collections import defaultdict

import numpy as np


class ConstrainedLPA:
    def __init__(self, adj, num_communities, node_threshold, terminate_delta):
        self.logger = logging.getLogger('constrained_lpa_single')

        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.num_communities = num_communities
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta

    def initialization(self):
        self.logger.info('initializing communities')

        random_nodes = np.arange(self.num_nodes)
        np.random.shuffle(random_nodes)
        self.communities = defaultdict(set)
        self.node_community = np.zeros(self.adj.shape[0])

        # each node use node is as its community label
        for community, nodes in enumerate(np.array_split(random_nodes, self.num_communities)):
            self.communities[community] = set(nodes)
            self.node_community[nodes] = community

    def community_detection(self, iterations=100):
        self.logger.info('detecting communities')

        communities = copy.deepcopy(self.communities)
        lpa_deltas = []

        # Currently, break when maximum iterations round achieves.
        for i in range(iterations):
            self.logger.info('iteration %s' % (i,))

            desire_move = self._determine_desire_move()
            sort_indices = np.flip(np.argsort(desire_move[:, 2]))
            candidate_nodes = defaultdict(list)

            # allocate nodes' community with descending order of colocate count
            for node in sort_indices:
                src_community = desire_move[node][0]
                dst_community = desire_move[node][1]

                if src_community != dst_community:
                    if len(self.communities[dst_community]) < self.node_threshold:
                        self.node_community[node] = dst_community
                        self.communities[dst_community].add(node)
                        self.communities[src_community].remove(node)

                        # reallocate the candidate nodes
                        candidate_nodes_cur = candidate_nodes[src_community]
                        while len(candidate_nodes_cur) != 0:
                            node_cur = candidate_nodes_cur[0]
                            src_community_cur = desire_move[node_cur][0]
                            dst_community_cur = desire_move[node_cur][1]

                            self.node_community[node_cur] = dst_community_cur
                            self.communities[dst_community_cur].add(node_cur)
                            self.communities[src_community_cur].remove(node_cur)

                            candidate_nodes[dst_community_cur].pop(0)
                            candidate_nodes_cur = candidate_nodes[src_community_cur]
                    else:
                        candidate_nodes[dst_community].append(node)
                # record the communities of each iteration, break the loop while communities are stable.

            delta = self._lpa_delta(communities, self.communities)
            lpa_deltas.append(delta)
            self.logger.info("%d" % delta)
            communities = copy.deepcopy(self.communities)
            if delta <= self.terminate_delta:
                break

        return self.communities, lpa_deltas

    def _determine_desire_move(self):
        desire_move = np.zeros([self.num_nodes, 3])
        desire_move[:, 0] = self.node_community

        for i in range(self.num_nodes):
            # neighbor_community = self.node_community[np.nonzero(self.adj[i])[0]]  # for non-bool adj
            neighbor_community = self.node_community[self.adj[i]]  # for bool adj
            unique_community, unique_count = np.unique(neighbor_community, return_counts=True)
            if unique_community.shape[0] == 0:
                continue
            max_indices = np.where(unique_count == np.max(unique_count))[0]

            if max_indices.size == 1:
                desire_move[i, 1] = unique_community[max_indices]
                desire_move[i, 2] = unique_count[max_indices]
            elif max_indices.size > 1:
                max_index = np.random.choice(max_indices)
                desire_move[i, 1] = unique_community[max_index]
                desire_move[i, 2] = unique_count[max_index]

        return desire_move

    def _lpa_delta(self, lpa_pre, lpa_cur):
        delta = 0.0
        for i in range(len(lpa_cur)):
            delta += len((lpa_cur[i] | lpa_pre[i]) - (lpa_cur[i] & lpa_pre[i]))

        return delta


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    adj = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]],
                   dtype=np.bool)

    num_communities = 2
    node_threshold = 3
    terminate_delta = 1

    lpa = ConstrainedLPA(adj, num_communities, node_threshold, terminate_delta)

    lpa.initialization()
    lpa.community_detection()
