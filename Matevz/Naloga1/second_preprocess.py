
import numpy as np
import pandas as pd



constructed_data_df = pd.read_csv("preprocessed.csv")
constructed_data = pd.read_csv("preprocessed.csv").values
# np.savetxt("to_country_and_edition_at_ixs.txt", np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), fmt="%s")

col_labels = pd.read_csv("col_labels.csv").values

row_labels = pd.read_csv("row_labels.csv").values

col_label_decomposition = pd.read_csv("col_label_decomposition.csv").values

from_count_voted_in_edition = pd.read_csv("from_count_voted_in_edition.csv")
# make countries the index
from_count_voted_in_edition.set_index("Unnamed: 0", inplace=True)

# print("from_count_voted_in_edition")
# print(from_count_voted_in_edition)


# print(constructed_data)

# find all rows which are all zeros and get a list of their row ixs
all_nan_rows_ixs = np.all(constructed_data_df.isnull().values, axis=1)
# print("all_nan_rows_ixs")
# print(all_nan_rows_ixs)

# remove these rows
constructed_data = constructed_data[~all_nan_rows_ixs]
row_labels = row_labels[~all_nan_rows_ixs]
from_count_voted_in_edition = from_count_voted_in_edition[~all_nan_rows_ixs]


# print("after zero removal")
# print("row_labels.T")
# print(row_labels.T)
# print("from_count_voted_in_edition")
# print(from_count_voted_in_edition)





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
    # Join the votes to Serbia
    from_count_voted_in_edition.loc["Serbia"] = from_count_voted_in_edition.loc["Serbia"] | from_count_voted_in_edition.loc["Yugoslavia"] | from_count_voted_in_edition.loc["Serbia & Montenegro"]
    
    serbia_ix = list(row_labels).index("Serbia")
    yugoslavia_ix = list(row_labels).index("Yugoslavia")
    serbia_and_montenegro_ix = list(row_labels).index("Serbia & Montenegro")
    constructed_data[serbia_ix] = constructed_data[serbia_ix] + constructed_data[yugoslavia_ix] + constructed_data[serbia_and_montenegro_ix]

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
    # do the same for North Macedonia and F.Y.R. Macedonia
    from_count_voted_in_edition.loc["North Macedonia"] = from_count_voted_in_edition.loc["North Macedonia"] | from_count_voted_in_edition.loc["F.Y.R. Macedonia"]

    north_macedonia_ix = list(row_labels).index("North Macedonia")
    fyr_macedonia_ix = list(row_labels).index("F.Y.R. Macedonia")
    constructed_data[north_macedonia_ix] = constructed_data[north_macedonia_ix] + constructed_data[fyr_macedonia_ix]

    # Remove the row of F.Y.R. Macedonia
    row_ixs_to_remove = [fyr_macedonia_ix]
    mask = np.array([i not in row_ixs_to_remove for i in range(len(row_labels))])
    constructed_data = constructed_data[mask]
    row_labels = row_labels[mask]
    from_count_voted_in_edition = from_count_voted_in_edition[mask]

    print("after:")
    print("from_count_voted_in_edition.loc[North Macedonia]")
    print(from_count_voted_in_edition.loc["North Macedonia"])



# print("After duplicate merging")
# print("row_labels.T")
# print(row_labels.T)
# print("from_count_voted_in_edition")
# print(from_count_voted_in_edition)


print("final constructed_data:")
print(constructed_data)


# make_into_reasonable_index
def ind(np_array_of_strings):
    new = []
    for i in np_array_of_strings:
        new.append(i[0])
    return new
    
# final constructed_data to pd with row and col labels
constructed_data_pd = pd.DataFrame(constructed_data, index=ind(row_labels), columns=ind(col_labels))
print("constructed_data_pd.columns")
print(constructed_data_pd.columns)


print("constructed_data_pd")
print(constructed_data_pd)
constructed_data_pd.to_csv("constructed_data.csv")

# # final from_count_voted_in_edition to pd with row and col labels
# print("before: from_count_voted_in_edition")
# print(from_count_voted_in_edition)
# from_count_voted_in_edition.reset_index(inplace=True)
# print("after: from_count_voted_in_edition")
# print(from_count_voted_in_edition)
# from_count_voted_in_edition.to_csv("from_count_voted_in_edition_refined.csv")











