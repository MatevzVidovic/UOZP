import math
import pandas as pd

num_of_clusters = 0
# TODO:
# - solve the closest_cluster nan problem (distance between cluster and all other clusters is nan)
# - try diferent linkage methods and distance computations
# - try different data preprocessing

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

        # index needed for visualization
        self.index = 10

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        self.index += 1
        values = []
        for cluster in clusters:
            cluster_values = []
            for char in str(cluster).split("'"):
                if char.isalpha():
                    cluster_values.append(data[char])
            values.append(cluster_values)

        dist_is_nan = True
        min_dist = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self.cluster_dist(values[i], values[j])
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
                    dist_is_nan = False
        
        if self.return_distances:
            if dist_is_nan:
                return clusters[0], clusters[1], math.nan
            return clusters[min_i], clusters[min_j], min_dist
        
        if dist_is_nan:
            return clusters[0], clusters[1], self.index
        return clusters[min_i], clusters[min_j], self.index
        
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
                clusters.append([distance, first, second, distance])
        global num_of_clusters
        num_of_clusters = self.index
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


def prepare_profile():
    df = pd.read_excel("eurovision_song_contest_1957_2023.xlsx")
    # Rename "Points      " column to "Points"
    df = df.rename(columns={"Points      ": "Points"})

    # Delete Edition column, because it's just a combinaton of Year and Edition
    df = df.drop(columns=["Edition"])

    # Delete all duplicates and delete the column "Duplicate"
    df = df[df.Duplicate != "x"]
    df = df.drop(columns=["Duplicate"])

    # Delete all rows which are not finals
    df = df[df["(semi-) final"] == "f"]
    # Delete "(semi-) final" column
    df = df.drop(columns=["(semi-) final"])

    # Delete rows with To country "Rest of the World" - not enough data
    df = df[df["To country"] != "Rest of the World"]

    # In 2016 voting system changed, from mixed system to two separate systems: jury and televoting and delete the column "Year"
    df_before_2016 = df[df.Year <= 2016]
    df_after_2016 = df[df.Year >= 2016]
    df_after_2016 = df_after_2016.drop(columns=["Year"])

    # Divide the data into two dataframes: one for jury votes and one for televoting and delete the column "Jury or Televoting"
    df_jury = df_after_2016[df_after_2016["Jury or Televoting"] == "J"]
    df_jury = df_jury.drop(columns=["Jury or Televoting"])
    df_televoting = df_after_2016[df_after_2016["Jury or Televoting"] == "T"]
    df_televoting = df_televoting.drop(columns=["Jury or Televoting"])
    df_before_2016 = df_before_2016.drop(columns=["Jury or Televoting"])

    # Calculate average points for each pair of From country and To country
    df_televoting_avg = df_televoting.groupby(["From country", "To country"]).mean()
    # df_televoting_avg.to_excel("eurovision_song_contest_2016_2023_avg.xlsx")

    # Count how many times each pair of From country and To country voted
    df_televoting_cnt = df_televoting.groupby(["From country", "To country"]).count()
    # df_televoting_cnt.to_excel("eurovision_song_contest_2016_2023_cnt.xlsx")

    # Make a matrix: From country are the names of the rows and To country are the names of the columns
    df_televoting_avg = df_televoting_avg.unstack(level=-1)
    # df_televoting_avg = df_televoting_avg.fillna(0)
    # df_televoting_avg.to_excel("eurovision_song_contest_2016_2023_matrix.xlsx")

    # Make a dict from the dataframe with From country as the key and values as a list of distances to To countries
    df_televoting_avg.columns = df_televoting_avg.columns.droplevel()
    data = df_televoting_avg.to_dict()
    for key in data:
        data[key] = list(data[key].values())
    
    return data

def run_hc(data):
    def average_linkage_w_euclidean(c1, c2):
        return average_linkage(c1, c2, euclidean_dist)
    
    def single_linkage_w_euclidean(c1, c2):
        return single_linkage(c1, c2, euclidean_dist)
    
    def complete_linkage_w_euclidean(c1, c2):
        return complete_linkage(c1, c2, euclidean_dist)
    
    def average_linkage_w_cosine(c1, c2):
        return average_linkage(c1, c2, cosine_dist)
    
    def single_linkage_w_cosine(c1, c2):
        return single_linkage(c1, c2, cosine_dist)
    
    def complete_linkage_w_cosine(c1, c2):
        return complete_linkage(c1, c2, cosine_dist)
    
    hc = HierarchicalClustering(cluster_dist=average_linkage_w_cosine)

    clusters = hc.run(data)
    return clusters

def flatten_clusters(clusters):
    flattened = []
    for item in clusters:
        if isinstance(item, list):
            flattened.extend(flatten_clusters(item))
        elif isinstance(item, str):
            flattened.append(item)
    return [x for x in flattened if isinstance(x, str)]

def create_dendrogram(clusters):
    # Get all countries in same order as in the clusters
    flattened = flatten_clusters(clusters)

    # Add spaces between countries for better visualization
    for i in range(len(flattened) - 1, 0, -1):
        flattened.insert(i, ' ')

    # Create list with height and middle of each country
    middle = [x for x in range(len(flattened))]
    height = [0 for x in range(len(flattened))]

    # Create dendrogram
    dendrogram = [flattened]

    clusters_str = str(clusters)  
    for i in range(11, num_of_clusters + 1):
        # Get countries in the next cluster
        splitted = clusters_str.split(str(i))
        cluster = splitted[1]
        if i == 48:
            a = 1
        countries = []
        for char in cluster.split("'"):
            if char.split()[0].isalpha():
                countries.append(char)

        # Get index of the first and last country
        first = flattened.index(countries[0])
        last = flattened.index(countries[-1])

        # Add 3 empty rows
        dendrogram.append([' ' for x in range(len(flattened))])
        dendrogram.append([' ' for x in range(len(flattened))])
        dendrogram.append([' ' for x in range(len(flattened))])

        # Draw '-'
        new_height = max(height[first], height[last]) + 3
        for i in range(min(height[first], height[last]), new_height):
            if i > height[first]:
                dendrogram[i][middle[first]] = "-"
            if i > height[last]:
                dendrogram[i][middle[last]] = "-"

        # Draw '|' and '+'
        new_middle = (middle[first] + middle[last]) // 2
        for i in range(min(middle[first], middle[last]), max(middle[first], middle[last]) + 1):
            dendrogram[new_height][i] = "|"
        dendrogram[new_height][new_middle] = "+"

        # Update height and middle
        for i in range(first, last + 1):
            height[i] = new_height
            middle[i] = new_middle
        
    return dendrogram

def draw_dendrogram(dendrogram):
    for row in range(len(dendrogram[0])):
        print(f'{dendrogram[0][row]:<15}', end="")
        for col in dendrogram[1:]:
            print(col[row], end="")
        print()

def dendrogram_scipy():
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram
    import matplotlib.pyplot as plt

    # Example linkage matrix (replace with your actual data)
    Z = np.array([
        [0, 1, 0.5, 2],   # Merge cluster 0 and 1 at distance 0.5, resulting in new cluster 2
        [2, 3, 0.8, 4],   # Merge cluster 2 and 3 at distance 0.8, resulting in new cluster 4
        # ... (more rows representing additional merges)
    ])

    # Plot the dendrogram
    plt.figure(figsize=(8, 6))
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()

    # # Load required modules
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from scipy.cluster.hierarchy import dendrogram

    # # Construct the graph/hierarchy
    # d           = { 0: [1, 'd'], 1: ['a', 'b', 'c'], 'a': [], 'b': [], 'c': [], 'd': []}
    # G           = nx.DiGraph(d)
    # nodes       = G.nodes()
    # leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
    # inner_nodes = [ n for n in nodes if G.out_degree(n) > 0 ]

    # # Compute the size of each subtree
    # subtree = dict( (n, [n]) for n in leaves )
    # for u in inner_nodes:
    #     children = set()
    #     node_list = list(d[u])
    #     while len(node_list) > 0:
    #         v = node_list.pop(0)
    #         children.add( v )
    #         node_list += d[v]

    #     subtree[u] = sorted(children & leaves)

    # inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last

    # # Construct the linkage matrix
    # leaves = sorted(leaves)
    # index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    # Z = []
    # k = len(leaves)
    # for i, n in enumerate(inner_nodes):
    #     children = d[n]
    #     x = children[0]
    #     for y in children[1:]:
    #         z = tuple(subtree[x] + subtree[y])
    #         i, j = index[tuple(subtree[x])], index[tuple(subtree[y])]
    #         Z.append([i, j, float(len(subtree[n])), len(z)]) # <-- float is required by the dendrogram function
    #         index[z] = k
    #         subtree[z] = list(z)
    #         x = z
    #         k += 1

    # # Visualize
    # dendrogram(Z, labels=leaves)
    # plt.show()


data = prepare_profile()
clusters = run_hc(data)
# print(clusters)
dendrogram = create_dendrogram(clusters)
draw_dendrogram(dendrogram)
# dendrogram_scipy()








