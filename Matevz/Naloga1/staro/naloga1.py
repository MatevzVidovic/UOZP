
import numpy as np
import pandas as pd

import math


constructed_data_df = pd.read_csv("constructed_data.csv")
constructed_data_df.set_index("Unnamed: 0", inplace=True)
constructed_data = constructed_data_df.values

col_labels = pd.read_csv("col_labels.csv").values

col_label_decomposition = pd.read_csv("col_label_decomposition.csv").values

print("constructed_data_df")
print(constructed_data_df)

print("constructed_data")
print(constructed_data)

# print("col_labels")
# print(col_labels)

print("col_label_decomposition")
print(col_label_decomposition)


print("\n" * 10)




from math import sqrt, isnan

from hw1 import euclidean_dist, manhattan_dist, \
    average_linkage, single_linkage, complete_linkage, \
    HierarchicalClustering, silhouette, silhouette_average



# # Dosn't work:
# # turn to a dictionary where row incicies are keys and row values are lists
# constructed_data_dict = constructed_data_df.to_dict(orient="dict")

constructed_data_dict = {}
for ix, label in enumerate(constructed_data_df.index):
    constructed_data_dict[label] = constructed_data[ix,:]



# print("constructed_data_dict")
# print(constructed_data_dict)




def cosine_dist(r1, r2):
    dot_prod_sum = 0
    r1_squard_L2_norm = 0
    r2_squard_L2_norm = 0


    for ix, first in enumerate(r1):
        second = r2[ix]

        if math.isnan(first) or math.isnan(second):
            continue
        
        dot_prod_sum += (first * second)
        r1_squard_L2_norm += first**2
        r2_squard_L2_norm += second**2

    product_of_lens = sqrt(r1_squard_L2_norm) * sqrt(r2_squard_L2_norm)
    
    # this means one of the vectors is a null vector
    # and this means it is both perpendicular and colinear
    # with the other vector. So We can choose 0, 1, or -1 as our cosine.
    # We say they are different, so our cosine is -1.
    if product_of_lens == 0:
        cos = -1
    else:
        cos = dot_prod_sum / product_of_lens


    return 1-cos



def in_ward_sum(cluster_np, centroid_np):
    sum = 0
    for ix, row in enumerate(cluster_np):
        sum += euclidean_dist(row, centroid_np)**2
    return sum

def average_linkage_ward_dist(c1, c2):
    """ Arguments c1 and c2 are 2d lists: [i-th memOfClust][j-th vec value of i-th member]
    """
    
    
    c1_np = np.array(c1)
    if c1_np.ndim == 1:
        c1_np = c1_np.reshape(-1, 1)
    c1_centroid = np.nanmean(c1_np, axis=0)

    c2_np = np.array(c2)
    if c2_np.ndim == 1:
        c2_np = c2_np.reshape(-1, 1)    
    c2_centroid = np.nanmean(c2_np, axis=0)

    together = np.concatenate((c1_np, c2_np), axis=0)
    together_centroid = np.nanmean(together, axis=0)


    if False:
        print("c1")
        print(c1)
        print("c1_np")
        print(c1_np)
        print("c1_np.ndim")
        print(c1_np.ndim)
        print("c2_np.ndim")
        print(c2_np.ndim)
        print("together.ndim")
        print(together.ndim)


    c1_sum = in_ward_sum(c1_np, c1_centroid)
    c2_sum = in_ward_sum(c2_np, c2_centroid)
    together_sum = in_ward_sum(together, together_centroid)

    sum = together_sum - (c1_sum + c2_sum)

    # closest clusters kept returning None and crashing towards the later stages.
    if math.isnan(sum) or sum == float("inf"):
        sum = float('+1E30')

    return sum




def average_linkage_w_manhattan(c1, c2):
        return average_linkage(c1, c2, manhattan_dist)


def average_linkage_w_cosine(c1, c2):
    return average_linkage(c1, c2, cosine_dist)


# hc = HierarchicalClustering(cluster_dist=average_linkage_ward_dist, return_distances=True)
hc = HierarchicalClustering(cluster_dist=average_linkage_w_cosine, return_distances=True)

# print("constructed_data_dict")
# print(constructed_data_dict)
clusters = hc.run(constructed_data_dict)

print("clusters")
print(clusters)