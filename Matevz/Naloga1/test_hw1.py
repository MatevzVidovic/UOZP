import unittest

from math import sqrt, isnan

from hw1 import euclidean_dist, manhattan_dist, \
    average_linkage, single_linkage, complete_linkage, \
    HierarchicalClustering, silhouette, silhouette_average


NAN = float("nan")


DATA = {'Albert': [22.0, 81.0, 32.0, 39.0, 21.0, 37.0, 46.0, 36.0, 99.0],
        'Branka': [91.0, 95.0, 65.0, 96.0, 89.0, 39.0, 11.0, 22.0, 29.0],
        'Cene': [51.0, 89.0, 21.0, 39.0, 100.0, 59.0, 100.0, 89.0, 27.0],
        'Dea': [9.0, 80.0, 18.0, 34.0, 61.0, 100.0, 90.0, 92.0, 8.0],
        'Edo': [93.0, 99.0, 39.0, 100.0, 12.0, 47.0, 17.0, 12.0, 63.0],
        'Franci': [49.0, 83.0, 17.0, 33.0, 92.0, 30.0, 98.0, 91.0, 73.0],
        'Helena': [91.0, 99.0, 97.0, 89.0, 49.0, 96.0, 81.0, 94.0, 69.0],
        'Ivan': [12.0, 69.0, 32.0, 14.0, 34.0, 12.0, 33.0, 48.0, 96.0],
        'Jana': [91.0, 80.0, 20.0, 10.0, 82.0, 93.0, 87.0, 91.0, 22.0],
        'Leon': [39.0, 100.0, 19.0, 29.0, 99.0, 31.0, 77.0, 79.0, 23.0],
        'Metka': [20.0, 91.0, 10.0, 15.0, 71.0, 99.0, 78.0, 93.0, 12.0],
        'Nika': [90.0, 60.0, 45.0, 34.0, 45.0, 20.0, 15.0, 5.0, 100.0],
        'Polona': [100.0, 98.0, 97.0, 89.0, 32.0, 72.0, 22.0, 13.0, 37.0],
        'Rajko': [14.0, 4.0, 15.0, 27.0, 61.0, 42.0, 51.0, 52.0, 39.0],
        'Stane': [9.0, 22.0, 8.0, 7.0, 100.0, 11.0, 92.0, 96.0, 29.0],
        'Zala': [85.0, 90.0, 100.0, 99.0, 45.0, 38.0, 92.0, 67.0, 21.0]}


CLUSTER_AVG_MAX = [[[['Rajko'], ['Stane']], [[['Franci'], [['Cene'], ['Leon']]],
                                             [['Jana'], [['Dea'], ['Metka']]]]],
                   [[['Nika'], [['Albert'], ['Ivan']]],
                    [[['Helena'], ['Zala']],
                     [['Branka'], [['Edo'], ['Polona']]]]]]


CLUSTER_MIN = [[[[['Dea'], ['Metka']], [['Jana'], [['Franci'],
                                                   [['Cene'], ['Leon']]]]],
                [['Rajko'], ['Stane']]], [[['Helena'], ['Zala']],
                                          [[['Branka'], [['Edo'], ['Polona']]],
                                           [['Nika'], [['Albert'], ['Ivan']]]]]]


def compare_trees(t1, t2):
    if len(t1) == 1 and len(t2) == 1:
        if t1[0].strip().lower() == t2[0].strip().lower():
            return True
        else:
            return False

    if len(t1) == 1 and len(t2) != 1:
        return False
    if len(t1) != 1 and len(t2) == 1:
        return False

    a0 = t1[0]
    b0 = t1[1]
    a1 = t2[0]
    b1 = t2[1]

    # also check distances if they are present at any of inputs
    if len(t1) == 3 or len(t2) == 3:
        if not (len(t1) == 3 and len(t2) == 3):
            raise RuntimeError("Some clusters seem to have distances, other not")
        if t1[2] != t2[2]:
            return False

    if compare_trees(a0, a1):
        left = True
    elif compare_trees(a0, b1):
        left = False
    else:
        return False

    if left:
        if compare_trees(b0, b1):
            return True
    else:
        if compare_trees(b0, a1):
            return True

    return False


def simple_distance(r1, r2):
    return abs(r1[0] - r2[0])


class ClusterCompare(unittest.TestCase):

    def assertEqualClosestClusters(self, cl1, cl2):
        # the order of clusters can be different, thus
        self.assertTrue(compare_trees(cl1[:-1], cl2[:-1]))
        self.assertEqual(cl1[-1], cl2[-1])

    def assertEqualFinalClusters(self, cl1, cl2):
        self.assertEqual(1, len(cl1))
        self.assertEqual(1, len(cl2))
        self.assertTrue(compare_trees(cl1[0], cl2[0]))


class HierarchicalClusteringTest(ClusterCompare):

    def test_euclidean_distance(self):
        dist = euclidean_dist([1,2,3], [1,3,5])
        self.assertAlmostEqual(dist, sqrt(0+1+4))
        dist = euclidean_dist([4,2,3], [1,3,5])
        self.assertAlmostEqual(dist, sqrt(9+1+4))
        dist = euclidean_dist(DATA["Polona"], DATA["Rajko"])
        self.assertAlmostEqual(dist, 175.803, places=2)

    def test_manhattan_distance(self):
        dist = manhattan_dist([1,2,3], [1,3,5])
        self.assertAlmostEqual(dist, 0+1+2)
        dist = manhattan_dist([4,2,3], [1,3,5])
        self.assertAlmostEqual(dist, 3+1+2)
        dist = manhattan_dist(DATA["Polona"], DATA["Rajko"])
        self.assertAlmostEqual(dist, 453.0, places=2)

    def test_single_linkage(self):
        d = single_linkage([[0], [1]], [[2], [4]], simple_distance)
        self.assertEqual(1, d)
        d = single_linkage([[0], [1]], [[2]], simple_distance)
        self.assertEqual(1, d)

    def test_complete_linkage(self):
        d = complete_linkage([[0], [1]], [[2], [4]], simple_distance)
        self.assertEqual(4, d)
        d = complete_linkage([[0], [1]], [[2]], simple_distance)
        self.assertEqual(2, d)

    def test_average_linkage(self):
        d = average_linkage([[0], [1]], [[2], [4]], simple_distance)
        self.assertEqual(2.5, d)
        d = average_linkage([[0], [1]], [[2]], simple_distance)
        self.assertEqual(1.5, d)

    def test_cluster_distance(self):
        c1 = [[0], [1]]
        c2 = [[2], [4]]
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, simple_distance)
        self.assertEqual(2.5, cluster_dist(c1, c2))

        c1 = [DATA[n] for n in ["Albert", "Branka", "Cene"]]
        c2 = [DATA[n] for n in ["Nika", "Polona"]]

        cluster_dist = lambda c1, c2: average_linkage(c1, c2, euclidean_dist)
        self.assertAlmostEqual(124.99, cluster_dist(c1, c2), places=2)

        cluster_dist = lambda c1, c2: single_linkage(c1, c2, euclidean_dist)
        self.assertAlmostEqual(75.94, cluster_dist(c1, c2), places=2)

        cluster_dist = lambda c1, c2: complete_linkage(c1, c2, euclidean_dist)
        self.assertAlmostEqual(165.86, cluster_dist(c1, c2), places=2)

    def test_closest_clusters(self):
        data = {"a": [1],
                "b": [2],
                "c": [5],
                "d": [7],
                "e": [12]}
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, simple_distance)
        hc = HierarchicalClustering(cluster_dist)
        closest = hc.closest_clusters(data, [["a"], ["b"], ["c"], ["d"]])
        self.assertEqualClosestClusters(closest, (["a"], ["b"], 1))
        closest = hc.closest_clusters(data, [["b"], ["c"], ["d"]])
        self.assertEqualClosestClusters(closest, (["c"], ["d"], 2))
        closest = hc.closest_clusters(data, [[["a"], ["b"]], ["c"], ["d"]])
        self.assertEqualClosestClusters(closest, (["c"], ["d"], 2))
        closest = hc.closest_clusters(data, [[["a"], ["b"]], [["c"], ["d"]], ["e"]])
        self.assertEqualClosestClusters(closest, ([["a"], ["b"]], [["c"], ["d"]], 4.5))

    def test_run_grades(self):
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, euclidean_dist)
        hc = HierarchicalClustering(cluster_dist)
        clusters = hc.run(DATA)
        self.assertEqualFinalClusters(clusters, [CLUSTER_AVG_MAX])


class HierarchicalClusteringUnknownsTest(ClusterCompare):

    small_data = {"a": [1, 2],
                  "b": [NAN, 1],
                  "c": [5, NAN],
                  "d": [NAN, 1],
                  "e": [12, 3]}

    def test_euclidean_distance(self):
        dist = euclidean_dist([1,2,NAN], [1,3,5])
        self.assertAlmostEqual(dist, 1.2247448713915891)
        dist = euclidean_dist([1,2,NAN], [1,3,NAN])
        self.assertAlmostEqual(dist, 1.2247448713915891)
        dist = euclidean_dist([NAN,2,NAN], [1,3,NAN])
        self.assertAlmostEqual(dist, sqrt(3))
        dist = euclidean_dist([1,NAN,NAN], [NAN,3,5])
        self.assertTrue(isnan(dist))

    def test_manhattan_distance(self):
        dist = manhattan_dist([1,2,NAN], [1,3,5])
        self.assertAlmostEqual(dist, 1.5)
        dist = manhattan_dist([1,2,NAN], [1,3,NAN])
        self.assertAlmostEqual(dist, 1.5)
        dist = manhattan_dist([NAN,2,NAN], [1,3,NAN])
        self.assertAlmostEqual(dist, 3)
        dist = manhattan_dist([1,NAN,NAN], [NAN,3,5])
        self.assertTrue(isnan(dist))

    def test_single_linkage(self):
        d = single_linkage([[0], [NAN]], [[2], [4]], simple_distance)
        self.assertEqual(2, d)
        d = single_linkage([[NAN], [1]], [[2], [4]], simple_distance)
        self.assertEqual(1, d)
        d = single_linkage([[0], [NAN]], [[2]], simple_distance)
        self.assertEqual(2, d)
        d = single_linkage([[NAN], [1]], [[NAN]], simple_distance)
        self.assertTrue(isnan(d))

    def test_complete_linkage(self):
        d = complete_linkage([[0], [NAN]], [[2], [4]], simple_distance)
        self.assertEqual(4, d)
        d = complete_linkage([[NAN], [1]], [[2], [4]], simple_distance)
        self.assertEqual(3, d)
        d = complete_linkage([[NAN], [1]], [[2]], simple_distance)
        self.assertEqual(1, d)
        d = complete_linkage([[NAN], [1]], [[NAN]], simple_distance)
        self.assertTrue(isnan(d))

    def test_average_linkage(self):
        d = average_linkage([[0], [NAN]], [[2], [4]], simple_distance)
        self.assertEqual(3, d)
        d = average_linkage([[NAN], [1]], [[2], [4]], simple_distance)
        self.assertEqual(2, d)
        d = average_linkage([[NAN], [1]], [[2]], simple_distance)
        self.assertEqual(1, d)
        d = average_linkage([[0], [NAN]], [[NAN], [4]], simple_distance)
        self.assertEqual(4, d)
        d = average_linkage([[NAN], [1]], [[NAN]], simple_distance)
        self.assertTrue(isnan(d))

    def test_cluster_distance(self):
        c1 = [[0], [1]]
        c2 = [[2], [NAN]]
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, simple_distance)
        self.assertEqual(1.5, cluster_dist(c1, c2))

        c1 = [self.small_data[n] for n in "abd"]

        cluster_dist = lambda c1, c2: average_linkage(c1, c2, manhattan_dist)
        self.assertEqual(20/3, cluster_dist(c1, [self.small_data["e"]]))

        cluster_dist = lambda c1, c2: average_linkage(c1, c2, manhattan_dist)
        self.assertEqual(8, cluster_dist(c1, [self.small_data["c"]]))

    def test_closest_clusters(self):
        data = self.small_data

        cluster_dist = lambda c1, c2: average_linkage(c1, c2, simple_distance)
        hc = HierarchicalClustering(cluster_dist)

        closest = hc.closest_clusters(data, [["a"], ["b"], ["c"], ["d"]])
        self.assertEqualClosestClusters(closest, (["a"], ["c"], 4))
        closest = hc.closest_clusters(data, [["a"], ["b"], ["c"]])
        self.assertEqualClosestClusters(closest, (["a"], ["c"], 4))
        closest = hc.closest_clusters(data, [[["a"], ["b"]], ["c"]])
        self.assertEqualClosestClusters(closest, ([["a"], ["b"]], ["c"], 4))
        closest = hc.closest_clusters(data, [[["a"], ["b"]], [["c"], ["d"]], ["e"]])
        self.assertEqualClosestClusters(closest, ([["a"], ["b"]], [["c"], ["d"]], 4))
        closest = hc.closest_clusters(data, [[["a"], ["b"]], [["c"], ["e"]], ["d"]])
        self.assertEqualClosestClusters(closest, ([["a"], ["b"]], [["c"], ["e"]], 7.5))

        cluster_dist = lambda c1, c2: average_linkage(c1, c2, manhattan_dist)
        hc = HierarchicalClustering(cluster_dist)

        closest = hc.closest_clusters(data, [["a"], ["b"], ["c"], ["d"], ["e"]])
        self.assertEqualClosestClusters(closest, (["b"], ["d"], 0))
        closest = hc.closest_clusters(data, [["a"], ["b"], ["c"]])
        self.assertEqualClosestClusters(closest, (["a"], ["b"], 2))
        closest = hc.closest_clusters(data, [[["b"], ["d"]], ["a"], ["c"], ["e"]])
        self.assertEqualClosestClusters(closest, ([["b"], ["d"]], ["a"], 2))
        closest = hc.closest_clusters(data, [[[["b"], ["d"]], ["a"]], ["c"], ["e"]])
        self.assertEqualClosestClusters(closest, ([[['b'], ['d']], ['a']], ['e'], 20/3))

    def test_run_small_data(self):
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, manhattan_dist)
        hc = HierarchicalClustering(cluster_dist)
        clusters = hc.run(self.small_data)
        self.assertEqualFinalClusters(clusters, [[[[[['b'], ['d']], ['a']], ['e']], ["c"]]])


class HierarchicalClusteringWithDistancesTest(ClusterCompare):

    small_data = {"a": [1, 2],
                  "b": [NAN, 1],
                  "c": [5, NAN],
                  "d": [NAN, 1],
                  "e": [12, 3]}

    def test_run_distances(self):
        cluster_dist = lambda c1, c2: average_linkage(c1, c2, manhattan_dist)
        hc = HierarchicalClustering(cluster_dist, return_distances=True)
        clusters = hc.run(self.small_data)
        self.assertEqual(1, len(clusters))
        self.assertEqual(11, clusters[0][-1])
        self.assertEqualFinalClusters(clusters,
            [[['c'], [['e'], [['a'], [['b'], ['d'], 0.0], 2.0], 6.666666666666667], 11.0]])


dataS4 = {"X": [1, 1],
          "Y": [0.9, 1],
          "Z": [1, 0],
          "Z1": [0.8, 0],
          "A": [0.5, 0.5],
          "A1": [0.6, 0.7],
          "A2": [0.4, 0.3],
          "A3": [0.8, 0.7],
          "B": [0.2, 0.1],
          "B2": [0.4, 0.5],
          }


dataS2 = {"X": [1, 1],
          "Y": [0.9, 1],
          "Z": [1, 0],
          "Z1": [0.8, 0]}


class SilhouetteTest(unittest.TestCase):

    def test_silhouette_basic(self):
        data = {"X":[1, 1],
                "Y": [0.9, 1],
                "Z": [1, 0]}

        s1 = silhouette("X", [["X", "Y"], ["Z"]], data)
        self.assertTrue(0.5 < s1 < 1)

    def test_silhouette_average_basic(self):
        data = {"X": [1, 1],
                "Y": [0.9, 1],
                "Z": [1, 0]}

        s1 = silhouette_average(data, [["X", "Y"], ["Z"]])  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y"]])  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X"]])  # še slabše skupine
        self.assertLess(s2, s1)
        self.assertLess(s3, s2)

    def test_silhouette1(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X"], ["Z", "Z1"]], data), 0.0)  # 0 by definition
        self.assertAlmostEqual(silhouette("Z", [["X"], ["Z", "Z1"]], data), 0.8, places=5)

    def test_silhouette2_mean(self):
        data = dataS2
        s1 = silhouette_average(data, [["X", "Y"], ["Z", "Z1"]])  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y", "Z1"]])  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X", "Z1"]])  # še slabše skupine
        self.assertAlmostEqual(s1, 0.851105768704104, places=5)
        self.assertAlmostEqual(s2, -0.42024432102251297, places=5)
        self.assertAlmostEqual(s3, -0.43077817395149687, places=5)

    def test_silhouette2(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X", "Y"], ["Z", "Z1"]], data), 0.9009804864072152, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Y"], ["Z", "Z1"]], data), 0.9004962809790011, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Y"], ["Z", "Z1"]], data), 0.800497515516439, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Y"], ["Z", "Z1"]], data), 0.8024487919137608, places=5)

        self.assertAlmostEqual(silhouette("X", [["X", "Z"], ["Y", "Z1"]], data), -0.44009804864072155, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Z"], ["Y", "Z1"]], data), -0.4502481404895006, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Z"], ["Y", "Z1"]], data), -0.39750621894395555, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Z"], ["Y", "Z1"]], data), -0.39312487601587404, places=5)

    def test_silhouette4(self):
        data = dataS4
        cl1 = [["X", "Y"], ["Z", "Z1"], ["A", "A1", "A2", "A3"], ["B", "B2"]]
        cl2 = [["Z", "Z1"], ["A", "A1", "A2", "A3"], ["X", "Y"], ["B", "B2"]]
        self.assertAlmostEqual(silhouette("X", cl1, data), 0.8393326749789924, places=5)
        self.assertAlmostEqual(silhouette("Y", cl1, data), 0.8215111850702933, places=5)
        self.assertAlmostEqual(silhouette("A", cl1, data), 0.10247919661515911, places=5)
        self.assertAlmostEqual(silhouette("B2", cl1, data), -0.42418101867909697, places=5)
        self.assertAlmostEqual(silhouette("X", cl2, data), 0.8393326749789924, places=5)
        self.assertAlmostEqual(silhouette("Y", cl2, data), 0.8215111850702933, places=5)
        self.assertAlmostEqual(silhouette("A", cl2, data), 0.10247919661515911, places=5)
        self.assertAlmostEqual(silhouette("B2", cl2, data), -0.42418101867909697, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
