#!/usr/bin/env python3

import unittest
import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
class TestSigmoid(unittest.TestCase):
    def test_zero(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)

    def test_neginf(self):
        self.assertAlmostEqual(sigmoid(float("-inf")), 0)
        
    def test_inf(self):
        self.assertAlmostEqual(sigmoid(float("inf")), 1)


        

def cluster_kmeans(X):
    from sklearn import cluster
    k_means = cluster.KMeans(n_clusters=10, random_state=10)
    labels = k_means.fit(X).labels_[::]
    #print(labels)
    return labels

        
class TestKMeans(unittest.TestCase):
        
    def test_clustering(self):
        from sklearn import cluster, datasets
        X, _ = datasets.load_boston(return_X_y=True)
        initial_result = cluster_kmeans(X)
        for x in range(0, 10):
            self.assertTrue(np.all(cluster_kmeans(X) == initial_result))
        

if __name__ == '__main__':
    unittest.main()
    