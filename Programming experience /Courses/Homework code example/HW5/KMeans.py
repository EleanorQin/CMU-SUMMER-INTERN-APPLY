import sys, math
import csv
import random
import time

# data: [1,2,3,4, ...]
class KMeans:
    def __init__(self, k, data):
        self._k = k
        self._data = data
        self.clusters = []
        self.labels = [0 for i in range(len(data))]

    def init_centroid(self, random_seed):
        random.seed(random_seed)
        idxs = random.sample(range(len(self._data)), self._k)
        centroid = []
        for i in idxs:
            centroid.append(self._data[i])
        self.centroids = centroid
        random.seed(int(time.time()))

    def distance(self, x1, x2):
        return abs(x1 - x2)

    def convergence(self, c1, c2):
        set1 = set(c1)
        set2 = set(c2)
        return (set1 == set2)

    def cluster(self):
        n = len(self._data)
        if_convergence = False
        while not if_convergence:
            for i in range(n):
                min_dist = math.inf
                for j in range(self._k):
                    dist = self.distance(self._data[i], self.centroids[j])
                    if dist <= min_dist:
                        min_dist = dist
                        self.labels[i] = j
            old_centroids = self.centroids.copy()
            for i in range(self._k):
                sum_cent = 0
                cnt_cent = 0
                for j in range(len(self.labels)):
                    if i == self.labels[j]:
                        sum_cent += self._data[j]
                        cnt_cent += 1
                self.centroids[i] = sum_cent/cnt_cent if cnt_cent != 0 else 0

            if_convergence = self.convergence(old_centroids, self.centroids)

if __name__ == '__main__':
    data = [16, 16, 4, 64, 4, 16, 8, 4, 4, 8]
    kmeans = KMeans(3, data)
    # kmeans.init_centroid(3)
    kmeans.cluster()
    print(data)
    print(kmeans.labels)

