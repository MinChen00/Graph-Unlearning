import logging
import copy

from tqdm import tqdm

import numpy as np
import cupy as np


class ConstrainedKmeans:
    def __init__(self, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=20):
        self.logger = logging.getLogger('constrained_kmeans')

        self.data_feat = data_feat
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration

    def initialization(self):
        centroids = np.random.choice(np.arange(self.data_feat.shape[0]), self.num_clusters, replace=False)
        self.centroid = {}
        for i in range(self.num_clusters):
            self.centroid[i] = self.data_feat[centroids[i].get()]

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        km_delta = []

        pbar = tqdm(total=self.max_iteration)
        pbar.set_description('Clustering')

        for i in range(self.max_iteration):
            self.logger.info('iteration %s' % (i,))

            self._node_reassignment()
            self._centroid_updating()

            # record the average change of centroids, if the change is smaller than a very small value, then terminate
            delta = self._centroid_delta(centroid, self.centroid)
            km_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)

            if delta <= self.terminate_delta:
                break
            self.logger.info("delta: %s" % delta)
        pbar.close()
        return self.clusters, km_delta

    def _node_reassignment(self):
        self.clusters = {}
        for i in range(self.num_clusters):
            self.clusters[i] = np.zeros(0, dtype=np.uint64)

        distance = np.zeros([self.num_clusters, self.data_feat.shape[0]])

        for i in range(self.num_clusters):
            distance[i] = np.sum(np.power((self.data_feat - self.centroid[i]), 2), axis=1)

        sort_indices = np.unravel_index(np.argsort(distance, axis=None), distance.shape)
        clusters = sort_indices[0]
        users = sort_indices[1]
        selected_nodes = np.zeros(0, dtype=np.int64)
        counter = 0

        while len(selected_nodes) < self.data_feat.shape[0]:
            cluster = int(clusters[counter])
            user = users[counter]
            if self.clusters[cluster].size < self.node_threshold:
                self.clusters[cluster] = np.append(self.clusters[cluster], np.array(int(user)))
                selected_nodes = np.append(selected_nodes, np.array(int(user)))

                # delete all the following pairs for the selected user
                user_indices = np.where(users == user)[0]
                a = np.arange(users.size)
                b = user_indices[user_indices > counter]
                remain_indices = a[np.where(np.logical_not(np.isin(a, b)))[0]]
                clusters = clusters[remain_indices]
                users = users[remain_indices]

            counter += 1

    def _centroid_updating(self):
        for i in range(self.num_clusters):
            self.centroid[i] = np.mean(self.data_feat[self.clusters[i].astype(int)], axis=0)

    def _centroid_delta(self, centroid_pre, centroid_cur):
        delta = 0.0
        for i in range(len(centroid_cur)):
            delta += np.sum(np.abs(centroid_cur[i] - centroid_pre[i]))

        return delta


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    data_feat = np.array([[1, 2],
                          [1, 3],
                          [1, 4],
                          [1, 5],
                          [10, 2],
                          [10, 3]])
    num_clusters = 2
    node_threshold = 3
    terminate_delta = 0.001

    cluster = ConstrainedKmeans(data_feat, num_clusters, node_threshold, terminate_delta)
    cluster.initialization()
    cluster.clustering()