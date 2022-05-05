# An implementation of `` Balanced Label Propagation for Partitioning MassiveGraphs'' (https://stanford.edu/~jugander/papers/wsdm13-blp.pdf)

import copy
import logging
from collections import defaultdict

import numpy as np
import cvxpy as cp
from scipy.stats import linregress


class ConstrainedLPABase:
    def __init__(self, adj, num_communities, node_threshold, terminate_delta):
        self.logger = logging.getLogger('constrained_lpa_base')

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

        for i in range(iterations):
            self.logger.info('iteration %s' % (i,))

            ## Step 1: calculate desired move
            desire_move = self._determine_desire_move()
            relocation = {}
            utility_func = {}

            ## Step 2: calculate parameters for linear programming problem
            for src_community in range(self.num_communities):
                for dst_community in range(self.num_communities):
                    move_node = desire_move[np.where(np.logical_and(desire_move[:, 1] == src_community, desire_move[:, 2] == dst_community))[0]]

                    if src_community != dst_community and move_node.size != 0:
                        move_node = move_node[np.flip(np.argsort(move_node[:, 3]))]
                        relocation[(src_community, dst_community)] = move_node

                        if move_node.shape[0] == 1:
                            utility_func[(src_community, dst_community)] = np.array([[0, move_node[0, 3]]])
                        else:
                            cum_sum = np.cumsum(move_node[:, 3])
                            utility_func_temp = np.zeros([move_node.shape[0] - 1, 2])
                            for k in range(move_node.shape[0] - 1):
                                utility_func_temp[k, 0], utility_func_temp[k, 1], _, _, _ = linregress([k, k+1], [cum_sum[k], cum_sum[k+1]])
                                utility_func[(src_community, dst_community)] = utility_func_temp

            ## Step 3: solve linear programming problem
            x = cp.Variable([self.num_communities, self.num_communities])
            z = cp.Variable([self.num_communities, self.num_communities])

            objective = cp.Maximize(cp.sum(z))
            constraints = []
            for src_community in range(self.num_communities):
                const = 0
                for dst_community in range(self.num_communities):
                    if (src_community, dst_community) in relocation:
                        if src_community == dst_community:
                            constraints.append(x[src_community, dst_community] == 0)
                            constraints.append(z[src_community, dst_community] == 0)
                        else:
                            ## Constraint 2 of Theorem 2
                            constraints.append(x[src_community, dst_community] >= 0)
                            constraints.append(x[src_community, dst_community] <= relocation[(src_community, dst_community)].shape[0])

                            ## Constraint 1 of Theorem 2
                            if (dst_community, src_community) in relocation:
                                const += x[src_community, dst_community] - x[dst_community, src_community]

                        ## Constraint 3 of Theorem 2
                        for utility_func_value in utility_func[(src_community, dst_community)]:
                            constraints.append(- utility_func_value[0] * x[src_community, dst_community] + z[src_community, dst_community] <= utility_func_value[1])

                    else:
                        constraints.append(x[src_community, dst_community] == 0)
                        constraints.append(z[src_community, dst_community] == 0)

                ## Constraint 1 of Theorem 2
                constraints.append(len(self.communities[src_community]) + const <= self.node_threshold)

            problem = cp.Problem(objective, constraints)
            problem.solve()

            ## Step 4: parse linear programming problem results
            if problem.status == 'optimal':
                x_value = np.floor(np.abs(x.value)).astype(np.int64)
                for src_community in range(self.num_communities):
                    for dst_community in range(self.num_communities):
                        if (src_community, dst_community) in relocation and x_value[src_community, dst_community] != 0:
                        # if (src_community, dst_community) in relocation:
                            relocation_temp = relocation[(src_community, dst_community)][:, 0].astype(np.int64)
                            move_node = relocation_temp[:x_value[src_community, dst_community] - 1]
                            if isinstance(move_node, np.int64):
                                self.communities[src_community].remove(move_node)
                                self.communities[dst_community].add(move_node)
                                self.node_community[move_node] = dst_community
                            else:
                                # move_node = set(move_node)
                                self.communities[src_community].difference_update(move_node)
                                self.communities[dst_community].update(move_node)
                                for node in move_node:
                                    self.node_community[node] = dst_community
            else:
                self.logger.info("No optimal solution, break!")
                break

            ## Check the number of moved nodes
            delta = self._lpa_delta(communities, self.communities)
            lpa_deltas.append(delta)
            self.logger.info("%d" % delta)
            communities = copy.deepcopy(self.communities)
            if delta <= self.terminate_delta:
                break

        return self.communities, lpa_deltas

    def _determine_desire_move(self):
        desire_move = []

        for i in range(self.num_nodes):
            # neighbor_community = self.node_community[np.nonzero(self.adj[i])[0]]  # for non-bool adj
            neighbor_community = self.node_community[self.adj[i]] # for bool adj
            unique_community, unique_count = np.unique(neighbor_community, return_counts=True)

            src_relocation = unique_count[np.where(unique_community == self.node_community[i])[0]]
            for community in unique_community:
                if community != self.node_community[i]:
                    dst_relocation = unique_count[np.where(unique_community == community)[0]]
                    if dst_relocation - src_relocation >= 0:
                        desire_move_temp = np.zeros(4)
                        desire_move_temp[0] = i
                        desire_move_temp[1] = self.node_community[i]
                        desire_move_temp[2] = community
                        desire_move_temp[3] = dst_relocation - src_relocation

                        desire_move.append(desire_move_temp)

        return np.stack(desire_move)

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

    lpa = ConstrainedLPABase(adj, num_communities, node_threshold, terminate_delta)

    lpa.initialization()
    lpa.community_detection()
