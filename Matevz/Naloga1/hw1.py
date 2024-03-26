import math
import numpy as np
import pandas as pd

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


def cosine_dist(r1, r2):
    from math import sqrt, isnan

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

    def closest_clusters(self, data, clusters, printout=False):
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
                    
                    if printout:
                        print("first_cluster")
                        print(first_cluster)
                        print("second_cluster")
                        print(second_cluster)
                    
                    currDist = self.cluster_dist(first_cluster, second_cluster)
                    
                    if printout:
                        print("currDist")
                        print(currDist)
                    
                    if not math.isnan(currDist) and currDist < closestDist:
                        closestDist = currDist
                        closestClustersIxs = [ixFirst, ixSecond]
            

        
        first = clusters[closestClustersIxs[0]]
        second = clusters[closestClustersIxs[1]]
        
        return (first, second, closestDist)
    

    def run(self, data, make_scipy_mat=False, printout=False):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        countries = list(data.keys())
        # print("countries")
        # print(countries)
        # print("data.keys()")
        # print(data.keys())

        # print("enumerate(countries)")
        # print(enumerate(countries))
        # print("enumerate(data.keys())")
        # print(enumerate(data.keys()))
        clust2ix = {str([name]): ix for ix, name in enumerate(countries)}
        curr_ix = len(countries)
        
        if printout:
            print("clust2ix")
            print(clust2ix)

        scipy_mat = []

        max_dist = -1e30

        all_cluster_combinations = [clusters.copy()]



        while len(clusters) >= 2:
            first, second, dist = self.closest_clusters(data, clusters)
            # update the "clusters" variable


            if not make_scipy_mat and self.return_distances:
                new_cluster = [first, second, dist]
            else:
                new_cluster = [first, second]

            if printout:
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

            if make_scipy_mat:

                clust2ix[str([first, second])] = curr_ix
                curr_ix += 1
                # print(clust2ix[str(first)])
                # print(clust2ix[str(second)])
                scipy_mat.append([clust2ix[str(first)], clust2ix[str(second)], dist, curr_ix])

                if dist > max_dist:
                    max_dist = dist

                all_cluster_combinations.append(clusters.copy())
            
            if printout:
                print("\n" * 5)
                
                print("next clusters")
                print(clusters)
            
        if make_scipy_mat:
            return clusters, scipy_mat, countries, max_dist, all_cluster_combinations
        
        return clusters


def clust_avg_dist(el, single_clust_list, data, dist_fn=cosine_dist):

    if len(single_clust_list) == 0:
        return 0

    el_vec = data[el]
    
    avg_dist = 0
    for i in single_clust_list:
        avg_dist += dist_fn(data[i], el_vec) / len(single_clust_list)
    
    return avg_dist

def silhouette(el, clusters, data, dist_fn=cosine_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    # print(el)
    # print(data)
    # print(clusters)

    el_clust_ix = None
    for ix, curr_clust in enumerate(clusters):
        if (el in curr_clust):
            el_clust_ix = ix
            break
    
    try:
        el_clust = clusters[el_clust_ix].copy()
        el_clust.remove(el)
    except:
        print("SILHOUETTE ERROR: el not in clusters.") 
        print("el")
        print(el)
        print("clusters")
        print(clusters)

    # By definition apparently
    if len(el_clust) <= 0:
        return 0.0

    inner_class_avg_dist = clust_avg_dist(el, el_clust, data, dist_fn)


    min_other_clust_dist = float("inf")
    for ix, curr_clust in enumerate(clusters):

        if ix == el_clust_ix:
            continue
        
        dist = clust_avg_dist(el, curr_clust, data, dist_fn)
        if dist < min_other_clust_dist:
            min_other_clust_dist = dist
    
    # if min_other_clust_dist != 0 or inner_class_avg_dist != 0:
    silhouette = (min_other_clust_dist - inner_class_avg_dist) / max(min_other_clust_dist, inner_class_avg_dist)
    # else:
        # silhouette = 0

    return silhouette


def silhouette_average(data, clusters, dist_fn=cosine_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    cases = list(data.keys())
    avg_silhouette = 0
    for case in cases:
        avg_silhouette += silhouette(case, clusters, data, dist_fn) / len(cases)
    
    return avg_silhouette






def preprocess(years_tuple, jury_or_tele="T", self_vote_filler=float("nan"), printout=False):


    import pandas as pd
    import numpy as np
    import math


    raw_data = pd.read_excel("eurovision_song_contest_1957_2023.xlsx")

    raw_data_np = raw_data.values

    if printout:
        print(raw_data.head())
        print(raw_data_np)



    # Finding unique values of each column

    for col in raw_data.columns:

        unique_values = raw_data[col].unique()

        if printout:
            print(f"Column: {col}")
            print(unique_values)
            print()


    if printout:
        print(5*"\n")



    # When was televoting first used?

    for ix in range(raw_data.shape[0]):
        if raw_data.at[ix, "Jury or Televoting"] == "T":
            if printout:
                print("Televoting first in:")
                print(raw_data.at[ix, "Year"])
            break



    """
    - Creating a dictionary of every edition to a df of its rows.
    - Finding out how many countries voted in each possible edition.
    """
    if False:
        editions = raw_data["Edition"].unique()

        edition2data_np = {}

        for edition in editions:
            is_curr_edition = raw_data_np[:, 2] == edition
            edition2data_np[edition] = raw_data_np[is_curr_edition, :]


        # np.set_printoptions(threshold=np.inf)
        print("edition2data_np[2015f]")
        print(edition2data_np["2015f"])
        np.set_printoptions(threshold=20)






        edition2num_of_voting_countries = {}

        for edition in editions:
            curr_data_np = edition2data_np[edition]
            voting_countries = np.unique(curr_data_np[:,5])

            edition2num_of_voting_countries[edition] = voting_countries.size

        print("edition2num_of_voting_countries")
        print(edition2num_of_voting_countries)



    if printout:
        print(5*"\n")



    acceptable_editions = [str(i) + "f" for i in range(years_tuple[0], years_tuple[1])]
    # print("acceptable_editions")
    # print(acceptable_editions)




    # Preparing a dictionary of data for all acceptable_editions
    acc_edition2data_np = {}

    for edition in acceptable_editions:
        is_curr_edition = raw_data_np[:, 2] == edition
        is_jury_or_tele = raw_data_np[:, 3] == jury_or_tele
        is_both = is_curr_edition & is_jury_or_tele
        # print("is_both")
        # print(is_both)

        acc_edition2data_np[edition] = raw_data_np[is_curr_edition, :]

    if printout:
        # np.set_printoptions(threshold=np.inf)
        print("acc_edition2data_np[2015f]")
        print(acc_edition2data_np["2015f"])
        # np.set_printoptions(threshold=20)





    from_countries = list(raw_data["From country"].unique())
    # to_countries = raw_data["To country"].unique()

    if printout:
        print(raw_data.columns)

    from_country_col_ix = int(list(raw_data.columns).index("From country"))
    to_country_col_ix = int(list(raw_data.columns).index("To country"))
    edition_col_ix = int(list(raw_data.columns).index("Edition"))
    points_ix = int(list(raw_data.columns).index("Points      "))


    to_country_and_edition_pairs = list()
    # This is meant for various grouping later:
    to_country_at_corresponding_ix = list()
    edition_at_corresponding_ix = list()

    for acc_edition in acceptable_editions:
        data_np = acc_edition2data_np[acc_edition]
        curr_to_countries = np.unique(data_np[:, to_country_col_ix])

        for country in curr_to_countries:
            country_edition_pair = country + acc_edition
            to_country_and_edition_pairs.append(country_edition_pair)
            to_country_at_corresponding_ix.append(country)
            edition_at_corresponding_ix.append(acc_edition)












    # fills it with NaNs
    constructed_data = pd.DataFrame(float('nan'), index=from_countries, columns=to_country_and_edition_pairs)

    if printout:
        # check if all entries in constructed_data are NaNs
        print("Is this all Nans?")
        print(constructed_data.isnull().values.all())
    # what does isnull() return? A boolean mask of the same shape as the DataFrame, True if the value is NaN, False otherwise.

    for edition, data_np in acc_edition2data_np.items():
        for row in data_np:
            
            from_country = row[from_country_col_ix]
            to_country = row[to_country_col_ix]
            to_country_and_edition_pair = to_country + edition

            # We usually make this NaN, because it is not a valid vote.
            if from_country.lower() == to_country.lower():
                constructed_data.loc[from_country, to_country_and_edition_pair] = self_vote_filler
                continue

            points = row[points_ix]

            if not math.isnan(points):
                constructed_data.loc[from_country, to_country_and_edition_pair] = points

    if printout:
        print("preprocessed.csv:")
        print(constructed_data)

    constructed_data.to_csv("on_the_fly_data/preprocessed.csv", index=False)
    # np.savetxt("to_country_and_edition_at_ixs.txt", np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), fmt="%s")

    col_labels = pd.DataFrame(to_country_and_edition_pairs)
    col_labels.to_csv("on_the_fly_data/col_labels.csv", index=False)

    row_labels = pd.DataFrame(from_countries)
    row_labels.to_csv("on_the_fly_data/row_labels.csv", index=False)

    col_label_decomposition = pd.DataFrame(np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)))
    col_label_decomposition.to_csv("on_the_fly_data/col_label_decomposition.csv", index=False)






    from_count_voted_in_edition = pd.DataFrame(False, index=from_countries, columns=acceptable_editions)

    # Editions where the country voted
    for edition, data_np in acc_edition2data_np.items():
        acc_edition2list_of_voting_countries = []
        
        curr_data = set(data_np[:, from_country_col_ix])

        for from_country in from_countries:
            if from_country in curr_data:
                from_count_voted_in_edition.at[from_country, edition] = True

    # save this to a file
    from_count_voted_in_edition.to_csv("on_the_fly_data/from_count_voted_in_edition.csv")

    if printout:
        print("from_count_voted_in_edition.csv:")
        print(from_count_voted_in_edition)




def second_preprocess(L2_normalize=False, printout=False):

    import numpy as np
    import pandas as pd
    import math



    constructed_data_df = pd.read_csv("on_the_fly_data/preprocessed.csv")
    constructed_data = pd.read_csv("on_the_fly_data/preprocessed.csv").values
    # np.savetxt("to_country_and_edition_at_ixs.txt", np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), fmt="%s")

    col_labels = pd.read_csv("on_the_fly_data/col_labels.csv").values

    row_labels = pd.read_csv("on_the_fly_data/row_labels.csv").values

    col_label_decomposition = pd.read_csv("on_the_fly_data/col_label_decomposition.csv").values

    from_count_voted_in_edition = pd.read_csv("on_the_fly_data/from_count_voted_in_edition.csv")
    # make countries the index
    from_count_voted_in_edition.set_index("Unnamed: 0", inplace=True)


    if printout:
        print("from_count_voted_in_edition:")
        print(from_count_voted_in_edition)

        print("preprocessed.csv:")
        print(constructed_data)

        print("\n" * 5)









    # make_into_reasonable_index
    def ind(np_array_of_strings):
        new = []
        for i in np_array_of_strings:
            new.append(i[0])
        return new



    # final constructed_data to pd with row and col labels
    before_serbia_data_pd = pd.DataFrame(constructed_data, index=ind(row_labels), columns=ind(col_labels))
    # keep only Serbia, Yugoslavia, and Serbia & Montenegro in the pd

    before_serbia_data_pd.loc[["Serbia", "Yugoslavia", "Serbia & Montenegro"],:].to_csv("on_the_fly_data/before_serbia_data.csv")
    # before_serbia_data_pd.to_csv("on_the_fly_data/before_serbia_data.csv")











    # Yugoslavia postane Serbia, ker gledamo od 1992 naprej in je takrat bila to predstavnica.
    # https://en.wikipedia.org/wiki/Yugoslavia_in_the_Eurovision_Song_Contest_1992
    # serbia and montenegro postane serbia, ker je večji del prebivalstva.

    # Check if Serbia ever voted when Yugoslavia or Serbia and Montenegro voted
    clash_pd_row = from_count_voted_in_edition.loc["Serbia"] & from_count_voted_in_edition.loc["Yugoslavia"] | from_count_voted_in_edition.loc["Serbia"] & from_count_voted_in_edition.loc["Serbia & Montenegro"] | from_count_voted_in_edition.loc["Yugoslavia"] & from_count_voted_in_edition.loc["Serbia & Montenegro"]

    # Check if any in clash_pd_row is True
    clash = clash_pd_row.any()

    # print("clash")
    # print(clash)

    if not clash:
        if printout:
            print("Serbia, Yugo, Serb and Montenegro never voted together")
        # Join the votes to Serbia
        from_count_voted_in_edition.loc["Serbia"] = from_count_voted_in_edition.loc["Serbia"] | from_count_voted_in_edition.loc["Yugoslavia"] | from_count_voted_in_edition.loc["Serbia & Montenegro"]
        
        serbia_ix = list(row_labels).index("Serbia")
        yugoslavia_ix = list(row_labels).index("Yugoslavia")
        serbia_and_montenegro_ix = list(row_labels).index("Serbia & Montenegro")

        length = constructed_data[serbia_ix].size
        for i in range(length):
            sum = 0
            if not math.isnan(constructed_data[serbia_ix][i]):
                sum += constructed_data[serbia_ix][i]
            if not math.isnan(constructed_data[yugoslavia_ix][i]):
                sum += constructed_data[yugoslavia_ix][i]
            if not math.isnan(constructed_data[serbia_and_montenegro_ix][i]):
                sum += constructed_data[serbia_and_montenegro_ix][i]

            constructed_data[serbia_ix][i] = sum

        # Remove the rows of Yugoslavia and Serbia and Montenegro
        row_ixs_to_remove = [yugoslavia_ix, serbia_and_montenegro_ix]
        mask = np.array([i not in row_ixs_to_remove for i in range(len(row_labels))])
        constructed_data = constructed_data[mask]
        row_labels = row_labels[mask]
        from_count_voted_in_edition = from_count_voted_in_edition[mask]






    # 'F.Y.R. Macedonia' postane 'North Macedonia' ker je to uradno ime od 2019 naprej
        
    # print("from_count_voted_in_edition.loc[North Macedonia]")
    # print(from_count_voted_in_edition.loc["North Macedonia"])

    # print("from_count_voted_in_edition.loc[F.Y.R. Macedonia]")
    # print(from_count_voted_in_edition.loc["F.Y.R. Macedonia"])

    # Check if Serbia ever voted when Yugoslavia or Serbia and Montenegro voted
    clash_pd_row = from_count_voted_in_edition.loc["North Macedonia"] & from_count_voted_in_edition.loc["F.Y.R. Macedonia"]

    # Check if any in clash_pd_row is True
    clash = clash_pd_row.any()

    # print("clash")
    # print(clash)

    if not clash:
        if printout:
            print("North Macedonia and F.Y.R. Macedonia never voted together")
        # do the same for North Macedonia and F.Y.R. Macedonia
        from_count_voted_in_edition.loc["North Macedonia"] = from_count_voted_in_edition.loc["North Macedonia"] | from_count_voted_in_edition.loc["F.Y.R. Macedonia"]

        north_macedonia_ix = list(row_labels).index("North Macedonia")
        fyr_macedonia_ix = list(row_labels).index("F.Y.R. Macedonia")

        length = constructed_data[north_macedonia_ix].size
        for i in range(length):
            sum = 0
            if not math.isnan(constructed_data[north_macedonia_ix][i]):
                sum += constructed_data[north_macedonia_ix][i]
            if not math.isnan(constructed_data[fyr_macedonia_ix][i]):
                sum += constructed_data[fyr_macedonia_ix][i]

            constructed_data[north_macedonia_ix][i] = sum

        # Remove the row of F.Y.R. Macedonia
        row_ixs_to_remove = [fyr_macedonia_ix]
        mask = np.array([i not in row_ixs_to_remove for i in range(len(row_labels))])
        constructed_data = constructed_data[mask]
        row_labels = row_labels[mask]
        from_count_voted_in_edition = from_count_voted_in_edition[mask]

        if printout:
            print("after:")
            print("from_count_voted_in_edition.loc[North Macedonia]")
            print(from_count_voted_in_edition.loc["North Macedonia"])


    if printout:
        print("After duplicate merging")
        print("constructed_data.shape")
        print(constructed_data.shape)
    # print("row_labels.T")
    # print(row_labels.T)
    # print("from_count_voted_in_edition")
    # print(from_count_voted_in_edition)

    # print("constructed_data after duplicate removal:")
    # print(constructed_data)







    """
    I moved this here, so I can split the analysis in 5 year periods at the start of the pipeline,
    and above countries don't fall off the map prematurely, causing errors"""

    # # find all rows which are all zeros and get a list of their row ixs
    # all_nan_rows_ixs = np.all(constructed_data_df.isnull().values, axis=1)

    # find if all elements in a row are null in constructed_data
    all_nan_rows_ixs = np.all(np.isnan(constructed_data), axis=1)

    # print("all_nan_rows_ixs")
    # print(all_nan_rows_ixs)

    # remove these rows
    constructed_data = constructed_data[~all_nan_rows_ixs]
    row_labels = row_labels[~all_nan_rows_ixs]
    from_count_voted_in_edition = from_count_voted_in_edition[~all_nan_rows_ixs]

    if printout:
        print("after zero removal")
        print("constructed_data.shape")
        print(constructed_data.shape)
    # print("row_labels.T")
    # print(row_labels.T)
    # print("from_count_voted_in_edition")
    # print(from_count_voted_in_edition)
    





    # L2 normalize the columns of constructed_data ignoring NaNs
    if L2_normalize:
        for i in range(constructed_data.shape[1]):
            col = constructed_data[:, i]
            col = col[~np.isnan(col)]

            if col.size == 0:
                continue

            col = col**2
            vec_length = math.sqrt(np.mean(col))
            constructed_data[:, i] = constructed_data[:, i] / vec_length
        














        
    # final constructed_data to pd with row and col labels
    constructed_data_pd = pd.DataFrame(constructed_data, index=ind(row_labels), columns=ind(col_labels))
    
    if printout:
        print("constructed_data_pd.columns")
        print(constructed_data_pd.columns)
        print("constructed_data.csv:")
        print(constructed_data_pd)

    constructed_data_pd.to_csv("on_the_fly_data/constructed_data.csv")

    # # final from_count_voted_in_edition to pd with row and col labels
    # print("before: from_count_voted_in_edition")
    # print(from_count_voted_in_edition)
    # from_count_voted_in_edition.reset_index(inplace=True)
    # print("after: from_count_voted_in_edition")
    # print(from_count_voted_in_edition)
    # from_count_voted_in_edition.to_csv("on_the_fly_data/from_count_voted_in_edition_refined.csv")


















def in_ward_sum(cluster_np, centroid_np):
    sum = 0
    for ix, row in enumerate(cluster_np):
        sum += euclidean_dist(row, centroid_np)**2
    return sum

def wards_method(c1, c2, printout=False):
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


    if printout:
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

def average_linkage_w_euclidian(c1, c2):
    return average_linkage(c1, c2, euclidean_dist)




def show_dendrogram(scipy_mat, countries, dist_cutoff, dist_name, linkage_name, str_bound_years):

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(scipy_mat, labels=countries, color_threshold=dist_cutoff)
    plt.title("Voting in years: " + str_bound_years + ". Using: " + linkage_name + ".")
    plt.xlabel("Country")
    plt.ylabel(dist_name)
    plt.show()




def list_of_keys_from_tree_of_lists(tree_of_lists):
        """Returns list of keys.
        Performs recursively."""
        
        if len(tree_of_lists) == 1 and not isinstance(tree_of_lists[0], list):
            return tree_of_lists
        
        curr_list = []


        for item in tree_of_lists:
            if isinstance(item, list):
                curr_list.extend(list_of_keys_from_tree_of_lists(item))
        
        # if not self.return_distances:        
        #     for i in tree_of_lists:
        #         curr_list.extend(self.list_of_keys_from_tree_of_lists(i))
        # else:
        #     for ix, item in tree_of_lists:
        #         if (ix+1) != len(tree_of_lists)
        
        return curr_list

def show_silhouette(all_cluster_combinations, constructed_data_dict, dist_fn, silho_dist_name, dist_linkage_name, str_bound_years, num_of_clusts=7, printout=False):
    """
    all_cluster_combinations: list of lists which are recursive structures of clusters.
    """
    chosen_clust_comb = None
    for curr_clust in all_cluster_combinations:
        if len(curr_clust) == num_of_clusts:
            chosen_clust_comb = curr_clust
            break
    
    chosen_clusts_flat = []
    for clust in chosen_clust_comb:
        chosen_clusts_flat.append(list_of_keys_from_tree_of_lists(clust))

    silho_for_each_clust = []
    for curr_clust_list in chosen_clusts_flat:
        curr_clust_silhos = []
        for country in curr_clust_list:
            curr_clust_silhos.append(silhouette(country, chosen_clusts_flat, constructed_data_dict, dist_fn))

        curr_clust_silhos.sort()
        silho_for_each_clust.append(curr_clust_silhos)
    
    
    if printout:
        print("chosen_clust_comb")
        print(chosen_clust_comb)
        print("chosen_clusts_flat")
        print(chosen_clusts_flat)
        print("silho_for_each_clust")
        print(silho_for_each_clust)

    # avg_silho = silhouette_average(constructed_data_dict, chosen_clusts_flat, dist_fn)


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm


    plt.xlim([-0.2, 1.0])
    
    razmik = 15
    # Inserting blank space between silhouette
    plt.ylim([0, 50 + (num_of_clusts+1) * razmik])

    y_lower = 0
    for ix, curr_clust_flat in enumerate(chosen_clusts_flat):

        color = cm.nipy_spectral(float(ix) / num_of_clusts)

        curr_clust_silhos = silho_for_each_clust[ix]
        # Plot each silhouette value
        for j in range(len(curr_clust_silhos)):
            plt.plot([0, curr_clust_silhos[j]], [j + y_lower, j + y_lower], color=color)
        
        y_lower += len(curr_clust_silhos)
        plt.text(0.0, y_lower, str(curr_clust_flat)+":")
        y_lower += razmik

    plt.title("Silhouette plot for " + dist_linkage_name + " for" + str_bound_years + ". Using: " + silho_dist_name + ".")
    plt.xlabel("Silhouette coefficient values")
    plt.xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.yticks([])  # Clear the yaxis labels
    plt.show()
    

    




    # print("list_of_keys_from_tree_of_lists(final_clusters)")
    # print(list_of_keys_from_tree_of_lists(final_clusters))


def naloga1(params, print_during=False, print_result=False, only_testing=False):

    import numpy as np
    import pandas as pd

    import math

    
    constructed_data_df = pd.read_csv("on_the_fly_data/constructed_data.csv")

    if only_testing:
        constructed_data_df = constructed_data_df.loc[1:15,:]
    
    constructed_data_df.set_index("Unnamed: 0", inplace=True)
    constructed_data = constructed_data_df.values

    col_labels = pd.read_csv("on_the_fly_data/col_labels.csv").values

    col_label_decomposition = pd.read_csv("on_the_fly_data/col_label_decomposition.csv").values

    if print_during:
        print("constructed_data_df")
        print(constructed_data_df)

        print("constructed_data")
        print(constructed_data)

        # print("col_labels")
        # print(col_labels)

        print("col_label_decomposition")
        print(col_label_decomposition)


        print("\n" * 10)



    

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



    









    hc = HierarchicalClustering(cluster_dist=average_linkage_w_cosine, return_distances=True)
    final_clusters, scipy_mat, countries, max_dist, all_cluster_combinations = hc.run(constructed_data_dict, make_scipy_mat=True)

    # print(max_dist)

    years_tuple = params["years_tuple"]
    show_dendrogram(scipy_mat, countries, params["cos_cutoff"], "Cosine distance", "average linkage", str(years_tuple))

    show_silhouette(all_cluster_combinations, constructed_data_dict, cosine_dist, "cosine_dist", "average_linkage_w_cosine", str(years_tuple),  num_of_clusts=params["num_of_clusts_cos"], printout=False)

    if print_result:
        print("average-cosine clusters")
        print(final_clusters)




    hc = HierarchicalClustering(cluster_dist=wards_method, return_distances=True)
    final_clusters, scipy_mat, countries, max_dist, all_cluster_combinations = hc.run(constructed_data_dict, make_scipy_mat=True)


    show_dendrogram(scipy_mat, countries, params["ward_cutoff"], "Ward's method", "Ward's method", str(years_tuple))

    show_silhouette(all_cluster_combinations, constructed_data_dict, euclidean_dist, "euclidean_dist", "Ward's method", str(years_tuple), num_of_clusts=params["num_of_clusts_ward"], printout=False)

    # print(max_dist)

    if print_result:
        print("ward's method clusters")
        print(final_clusters)



    
    # hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan, return_distances=True)
    # final_clusters, scipy_mat, countries, max_dist, all_cluster_combinations = hc.run(constructed_data_dict, make_scipy_mat=True)

    # show_dendrogram(scipy_mat, countries, 1970, "Manhattan distance", "average linkage", "1992-2023")

    # show_silhouette(all_cluster_combinations, constructed_data_dict, manhattan_dist, num_of_clusts=10, printout=True)

    # print(max_dist)

    # if print_result:
    #     print("average_linkage_w_manhattan clusters")
    #     print(final_clusters)
    





    # hc = HierarchicalClustering(cluster_dist=average_linkage_w_euclidian, return_distances=True)
    # final_clusters, scipy_mat, countries, max_dist, all_cluster_combinations = hc.run(constructed_data_dict, make_scipy_mat=True)

    # print(max_dist)

    # show_dendrogram(scipy_mat, countries, 0.5, "Euclidean distance", "average linkage", "1992-2023")

    # show_silhouette(all_cluster_combinations, constructed_data_dict, euclidean_dist, printout=True)

    # if print_result:
    #     print("average-euclidean clusters")
    #     print(final_clusters)





    

if __name__ == "__main__":


    import os 
   
    # # Directory 
    # directory = "GeeksForGeeks"
    
    # # Parent Directory path 
    # parent_dir = "/home/User/Documents"
    
    # # Path 
    # path = os.path.join(parent_dir, directory) 
    
    # # Create the directory 
    # # 'GeeksForGeeks' in 
    # # '/home / User / Documents'
    path = "on_the_fly_data" 
    try:
        os.mkdir(path)
    except OSError as error:
        _ = "do nothing, all working as intended." 

    # years_tuple = (1992, 2023)
    # years_tuple = (2016, 2023)
    params_dict_1 = {
        "years_tuple" : (1992, 2023),
        
        "L2_normalize": False,

        "jury_or_tele": "T",
        "self_vote_filler":float("nan"),

        "num_of_clusts_cos" : 8,
        "cos_cutoff": 0.467,
        "num_of_clusts_ward" : 11,
        "ward_cutoff": 9450
    }

    params_dict_2 = {
        "years_tuple" : (2016, 2023),

        "L2_normalize": False,

        "jury_or_tele": "T",
        "self_vote_filler":float("nan"),

        "num_of_clusts_cos" : 5,
        "cos_cutoff": 0.385,
        "num_of_clusts_ward" : 7,
        "ward_cutoff": 1800
    }

    params_dict_3 = {
        "years_tuple" : (1992, 2023),
        
        "L2_normalize": True,

        "jury_or_tele": "T",
        "self_vote_filler":float("nan"),

        "num_of_clusts_cos" : 10,
        "cos_cutoff": 0.467,
        "num_of_clusts_ward" : 9,
        "ward_cutoff": 9450
    }

    params_dict_4 = {
        "years_tuple" : (2016, 2023),

        "L2_normalize": True,

        "jury_or_tele": "T",
        "self_vote_filler":float("nan"),

        "num_of_clusts_cos" : 5,
        "cos_cutoff": 0.385,
        "num_of_clusts_ward" : 7,
        "ward_cutoff": 1800
    }
    
    params = params_dict_2

    preprocess(params["years_tuple"], params["jury_or_tele"], params["self_vote_filler"])
    second_preprocess(params["L2_normalize"])
    naloga1(params, print_result=False, only_testing=False)



    # data = {"a": [1, 2],
    #         "b": [2, 3],
    #         "c": [5, 5]}

    # def average_linkage_w_manhattan(c1, c2):
    #     return average_linkage(c1, c2, manhattan_dist)

    # hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan)
    # clusters = hc.run(data)
    # print(clusters)  # [[['c'], [['a'], ['b']]]] (or equivalent)

    # hc = HierarchicalClustering(cluster_dist=average_linkage_w_manhattan,
    #                             return_distances=True)
    # clusters = hc.run(data)
    # print(clusters)  # [[['c'], [['a'], ['b'], 2.0], 6.0]] (or equivalent)
