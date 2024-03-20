import math
import pandas as pd

def manhattan_dist(r1, r2):
    # """ Arguments r1 and r2 are lists of numbers """
    count_nan = 0
    dist = 0
    for x, y in zip(r1, r2):
        if math.isnan(x) or math.isnan(y):
            count_nan += 1
        else:
            dist += abs(x - y)
    
    if count_nan == len(r1):
        return math.nan
    else:
        return dist + dist / (len(r1) - count_nan) * count_nan

def euclidean_dist(r1, r2):
    count_nan = 0
    dist = 0
    for x, y in zip(r1, r2):
        if math.isnan(x) or math.isnan(y):
            count_nan += 1
        else:
            dist += (x - y) ** 2
    
    if count_nan == len(r1):
        return math.nan
    else:
        return math.sqrt(dist + dist / (len(r1) - count_nan) * count_nan)

def cosine_dist(r1, r2):
    count_nan = 0
    dot_product = 0
    norm1 = 0
    norm2 = 0
    for x, y in zip(r1, r2):
        if math.isnan(x) or math.isnan(y):
            count_nan += 1
        else:
            dot_product += x * y
            norm1 += x ** 2
            norm2 += y ** 2

    if count_nan == len(r1):
        return math.nan
    else:
        return 1 - dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    min_dist = float("inf")
    for x in c1:
        for y in c2:
            dist = distance_fn(x, y)
            if not math.isnan(dist) and dist < min_dist:
                min_dist = dist

    if min_dist == float("inf"):
        return math.nan
    else:
        return min_dist

def complete_linkage(c1, c2, distance_fn):
    max_dist = float("-inf")
    for x in c1:
        for y in c2:
            dist = distance_fn(x, y)
            if not math.isnan(dist) and dist > max_dist:
                max_dist = dist

    if max_dist == float("-inf"):
        return math.nan
    else:
        return max_dist

def average_linkage(c1, c2, distance_fn):
    sum_dist = 0
    count = 0
    for x in c1:
        for y in c2:
            dist = distance_fn(x, y)
            if not math.isnan(dist):
                sum_dist += dist
                count += 1
    
    if count == 0:
        return math.nan
    else:
        return sum_dist / count


class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        values = []
        for cluster in clusters:
            cluster_values = []
            for char in str(cluster).split("'"):
                if char.isalpha():
                    cluster_values.append(data[char])
            values.append(cluster_values)

        min_dist = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self.cluster_dist(values[i], values[j])
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j

        return clusters[min_i], clusters[min_j], min_dist
        
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
            first, second, distance = self.closest_clusters(data, clusters)
            # update the "clusters" variable
            clusters.remove(first)
            clusters.remove(second)
            if self.return_distances:
                clusters.append([first, second, distance])
            else:
                clusters.append([first, second])      
        return clusters


def silhouette(el1, clusters, data):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
   
    b = float("inf")

    for cluster in clusters:
        dist = 0
        for el2 in cluster:
            if el1 != el2:
                dist += euclidean_dist(data[el1], data[el2])
        
        if el1 in cluster:
            if len(cluster) > 1:
                a = dist / (len(cluster) - 1)
            else:
                return 0
        else:
            if dist / len(cluster) < b:
                b = dist / len(cluster)

    return (b - a) / max(a, b)

def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    silhouette_sum = 0
    for el in data:
        silhouette_sum += silhouette(el, clusters, data)

    return silhouette_sum / len(data)


# if __name__ == "__main__":

#     data = {"a": [1, 2],
#             "b": [2, 3],
#             "c": [5, 5]}

#     def average_linkage_w_manhattan(c1, c2):
#         return average_linkage(c1, c2, manhattan_dist)

#     hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
#     clusters = hc.run(data)
#     print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

#     hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
#                                 return_distances=True)
#     clusters = hc.run(data)
#     print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)

def prepare_profile():
    df = pd.read_excel("eurovision_song_contest_1957_2023.xlsx")

    # Delete Edition column, because it's just a combinaton of Year and Edition
    df = df.drop(columns=["Edition"])

    # Delete all duplicates
    df = df[df.Duplicate != "x"]
    # Delete Duplicate column
    df = df.drop(columns=["Duplicate"])

    # TRY WITH POINTS AVERAGE AND TIME PERIODS
    # In 2016 voting system changed, from mixed system to two separate systems: jury and televoting
    df_before_2016 = df[df.Year <= 2016]
    df_after_2016 = df[df.Year >= 2016]

    # Divide the data into two dataframes: one for jury votes and one for televoting
    df_jury = df_after_2016[df_after_2016["Jury or Televoting"] == "J"]
    df_jury = df_jury.drop(columns=["Jury or Televoting"])
    df_televoting = df_after_2016[df_after_2016["Jury or Televoting"] == "T"]
    df_televoting = df_televoting.drop(columns=["Jury or Televoting"])
    df_before_2016 = df_before_2016.drop(columns=["Jury or Televoting"])

    # solve problem with semifinales
    # maybe start with hierarchical clustering after 2016 (first telvoting and then jury) and just use average number of points for each country
    # From country is the name of the "point", To country (use a dict of countries) and points are the values
    
    # Also ask chatgpt how would it solve this problem

    # TRY WITH PCA

prepare_profile()