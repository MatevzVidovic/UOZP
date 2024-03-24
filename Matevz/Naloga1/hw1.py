import math

def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    sum = 0
    nan_count = 0

    for ix, first in enumerate(r1):
        second = r2[ix]

        if math.isnan(first) or math.isnan(second):
            nan_count += 1
        else:
            sum += abs(first - second)

    # if all cases were nan
    if nan_count == len(r1):
        return float("nan")
    
    sum = sum * (len(r1) / (len(r1) - nan_count))


    return sum


def euclidean_dist(r1, r2):
    sum = 0
    nan_count = 0

    for ix, first in enumerate(r1):
        second = r2[ix]

        if math.isnan(first) or math.isnan(second):
            nan_count += 1
        else:
            sum += (first - second)**2

    # if all cases were nan
    if nan_count == len(r1):
        return float("nan")

    sum = sum * (len(r1) / (len(r1) - nan_count))
    sum = sum**(1/2)

    return sum


def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are 2d lists: [i-th memOfClust][j-th vec value of i-th member]

        Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""

    smallestDist = float("inf")

    count_nan = 0
    count_all = 0

    for listInFirst in c1:
        for listInSecond in c2:
            currDist = distance_fn(listInFirst, listInSecond)
            count_all += 1
            if math.isnan(currDist):
                count_nan += 1
            elif currDist < smallestDist:
                smallestDist = currDist
    
    
    if count_nan == count_all:
        return float("nan")
    
    return smallestDist
    


def complete_linkage(c1, c2, distance_fn):

    largestDist = 0

    count_nan = 0
    count_all = 0

    for listInFirst in c1:
        for listInSecond in c2:
            currDist = distance_fn(listInFirst, listInSecond)
            count_all += 1
            if math.isnan(currDist):
                count_nan += 1
            elif currDist > largestDist:
                largestDist = currDist
    
    if count_nan == count_all:
        return float("nan")
    
    return largestDist




def average_linkage(c1, c2, distance_fn):

    cumDist = 0

    # print(10*"&&&&&&&&&&\n")
    # print(c1)
    # print(c2)

    count_nan = 0
    count_all = 0

    for listInFirst in c1:
        for listInSecond in c2:
            count_all += 1
            currDist = distance_fn(listInFirst, listInSecond)
            if math.isnan(currDist):
                count_nan += 1
            else:
                cumDist += currDist
    
    if count_nan == count_all:
        return float("nan")

    avgDist = cumDist / (count_all - count_nan)

    # print(avgDist)
    return avgDist







class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def list_of_keys_from_tree_of_lists(self, tree_of_lists):
        """Returns list of keys.
        Performs recursively."""
        
        if len(tree_of_lists) == 1 and not isinstance(tree_of_lists[0], list):
            return tree_of_lists
        
        curr_list = []


        for item in tree_of_lists:
            if isinstance(item, list):
                curr_list.extend(self.list_of_keys_from_tree_of_lists(item))
        
        # if not self.return_distances:        
        #     for i in tree_of_lists:
        #         curr_list.extend(self.list_of_keys_from_tree_of_lists(i))
        # else:
        #     for ix, item in tree_of_lists:
        #         if (ix+1) != len(tree_of_lists)
        
        return curr_list

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        closestClustersIxs = None
        closestDist = float("inf")


        # 3D list.   [cluster_ix][key_ix][key_value]
        data_clusters_lists_of_lists = []
        for curr_clust in clusters:
            curr_cluster_keys = self.list_of_keys_from_tree_of_lists(curr_clust)
            curr_data = [data[key] for key in curr_cluster_keys]
            data_clusters_lists_of_lists.append(curr_data)

        for ixFirst in range(len(data_clusters_lists_of_lists)):
            for ixSecond in range(ixFirst+1, len(data_clusters_lists_of_lists)):
                first_cluster = data_clusters_lists_of_lists[ixFirst]
                second_cluster = data_clusters_lists_of_lists[ixSecond]
                currDist = self.cluster_dist(first_cluster, second_cluster)
                
                if not math.isnan(currDist) and currDist < closestDist:
                    closestDist = currDist
                    closestClustersIxs = [ixFirst, ixSecond]
        


        if closestClustersIxs is None:
            # 3D list.   [cluster_ix][key_ix][key_value]
            data_clusters_lists_of_lists = []
            for curr_clust in clusters:
                curr_cluster_keys = self.list_of_keys_from_tree_of_lists(curr_clust)
                curr_data = [data[key] for key in curr_cluster_keys]
                data_clusters_lists_of_lists.append(curr_data)
            # print("data_clusters_lists_of_lists")
            # print(data_clusters_lists_of_lists)

            for ixFirst in range(len(data_clusters_lists_of_lists)):
                for ixSecond in range(ixFirst+1, len(data_clusters_lists_of_lists)):
                    first_cluster = data_clusters_lists_of_lists[ixFirst]
                    second_cluster = data_clusters_lists_of_lists[ixSecond]
                    print("first_cluster")
                    print(first_cluster)
                    print("second_cluster")
                    print(second_cluster)
                    currDist = self.cluster_dist(first_cluster, second_cluster)
                    print("currDist")
                    print(currDist)
                    
                    if not math.isnan(currDist) and currDist < closestDist:
                        closestDist = currDist
                        closestClustersIxs = [ixFirst, ixSecond]
            

        
        first = clusters[closestClustersIxs[0]]
        second = clusters[closestClustersIxs[1]]
        
        return (first, second, closestDist)
    

    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, dist = self.closest_clusters(data, clusters)
            # update the "clusters" variable


            if self.return_distances:
                new_cluster = [first, second, dist]
            else:
                new_cluster = [first, second]

            print("\n" * 5)

            print("clusters")
            print(clusters)
            
            print("first")
            print(first)
            print("second")
            print(second)
            clusters.remove(first)
            clusters.remove(second)
            clusters.append(new_cluster)
            
            print("\n" * 5)
            
            print("next clusters")
            print(clusters)
            

        return clusters


def clust_avg_dist(el, single_clust_list, data):

    if len(single_clust_list) == 0:
        return 0

    el_vec = data[el]
    
    avg_dist = 0
    for i in single_clust_list:
        avg_dist += euclidean_dist(data[i], el_vec) / len(single_clust_list)
    
    return avg_dist

def silhouette(el, clusters, data):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    # print(el)
    # print(data)
    # print(clusters)

    el_clust_ix = 0
    for ix, curr_clust in enumerate(clusters):
        if (el in curr_clust):
            el_clust_ix = ix
            break
    
    el_clust = clusters[el_clust_ix].copy()
    el_clust.remove(el)

    # By definition apparently
    if len(el_clust) <= 0:
        return 0.0

    inner_class_avg_dist = clust_avg_dist(el, el_clust, data)


    min_other_clust_dist = float("inf")
    for ix, curr_clust in enumerate(clusters):

        if ix == el_clust_ix:
            continue
        
        dist = clust_avg_dist(el, curr_clust, data)
        if dist < min_other_clust_dist:
            min_other_clust_dist = dist
    
    # if min_other_clust_dist != 0 or inner_class_avg_dist != 0:
    silhouette = (min_other_clust_dist - inner_class_avg_dist) / max(min_other_clust_dist, inner_class_avg_dist)
    # else:
        # silhouette = 0

    return silhouette


def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    cases = list(data.keys())
    avg_silhouette = 0
    for case in cases:
        avg_silhouette += silhouette(case, clusters, data) / len(cases)
    
    return avg_silhouette


if __name__ == "__main__":

    data = {"a": [1, 2],
            "b": [2, 3],
            "c": [5, 5]}

    def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

    hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
                                return_distances=True)
    clusters = hc.run(data)
    print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
