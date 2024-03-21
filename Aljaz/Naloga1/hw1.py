import math
import pandas as pd

# TODO:
# - draw the dendrogram
# - solve the closest_sluster nan problem (distance between cluster and all other clusters is nan)
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
    
    hc = HierarchicalClustering(cluster_dist=average_linkage_w_euclidean)
    clusters = hc.run(data)
    # print(clusters)
    return clusters

def flatten_clusters(clusters):
    flattened = []
    for item in clusters:
        if isinstance(item, list):
            flattened.extend(flatten_clusters(item))
        else:
            flattened.append(item)
    return flattened

def create_dendrogram(clusters):
    flattened = flatten_clusters(clusters)
    flattened = [x for x in flattened if isinstance(x, str)]

    for i in range(len(flattened) - 1, 0, -1):
        flattened.insert(i, ' ')

    middle = [x for x in range(len(flattened))]
    height = [0 for x in range(len(flattened))]
    dendrogram = [flattened]

    clusters_str = str(clusters)
    for i in range(11, 52):
        splitted = clusters_str.split(str(i))
        cluster = splitted[1]



        


        countries = []
        for char in cluster.split("'"):
            if char.isalpha():
                countries.append(char)

        dendrogram.append([' ' for x in range(len(flattened))])
        dendrogram.append([' ' for x in range(len(flattened))])
        dendrogram.append([' ' for x in range(len(flattened))])

        # Get index of the first and last country
        first = flattened.index(countries[0])
        last = flattened.index(countries[-1])

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

        '''

        Russia  --|
                  +-----|
        Ukraine --|     |
                        |
        Albania --|     +
                  +--|  |
        Croatia --|  |  |
                     +--|
        Serbia  --|  |
                  +--|
        Latvia  --|
        
        '''
        # break
    return dendrogram

def draw_dendrogram(dendrogram):
    for row in range(len(dendrogram[0])):
        print(f'{dendrogram[0][row]:<15}', end="")
        for col in dendrogram[1:]:
            print(col[row], end="")
        print()


data = prepare_profile()
clusters = run_hc(data)
# print(clusters)
# dendrogram(clusters)
dendrogram = create_dendrogram(clusters)
draw_dendrogram(dendrogram)








