

import pandas as pd
import numpy as np
import math

file_path = "path_to_your_file.xlsx"

raw_data = pd.read_excel("eurovision_song_contest_1957_2023.xlsx")
print(raw_data.head())

raw_data_np = raw_data.values
print(raw_data_np)



# Finding unique values of each column

for col in raw_data.columns:

    unique_values = raw_data[col].unique()

    print(f"Column: {col}")
    print(unique_values)
    print()



print(5*"\n")



# When was televoting first used?

print("Televoting first in:")
for ix in range(raw_data.shape[0]):
    if raw_data.at[ix, "Jury or Televoting"] == "T":
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




print(5*"\n")




acceptable_editions = [str(i) + "f" for i in range(1992,2024)]
# print("acceptable_editions")
# print(acceptable_editions)




# Preparing a dictionary of data for all acceptable_editions
acc_edition2data_np = {}

for edition in acceptable_editions:
    is_curr_edition = raw_data_np[:, 2] == edition
    acc_edition2data_np[edition] = raw_data_np[is_curr_edition, :]


# np.set_printoptions(threshold=np.inf)
print("acc_edition2data_np[2015f]")
print(acc_edition2data_np["2015f"])
# np.set_printoptions(threshold=20)





from_countries = list(raw_data["From country"].unique())
# to_countries = raw_data["To country"].unique()


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



# fills it with zeros
constructed_data = pd.DataFrame(0, index=from_countries, columns=to_country_and_edition_pairs)

print("Is this all zeros?")
print((constructed_data == 0).all().all())

for edition, data_np in acc_edition2data_np.items():
    for row in data_np:
        
        from_country = row[from_country_col_ix]
        to_country = row[to_country_col_ix]

        to_country_and_edition_pair = to_country + edition
        points = row[points_ix]

        if not math.isnan(points):
            constructed_data.loc[from_country, to_country_and_edition_pair] = points


print(constructed_data)

constructed_data.to_csv("preprocessed.csv", index=False)
# np.savetxt("to_country_and_edition_at_ixs.txt", np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), fmt="%s")

col_labels = pd.DataFrame(to_country_and_edition_pairs)
col_labels.to_csv("col_labels.csv", index=False)

row_labels = pd.DataFrame(to_country_and_edition_pairs)
row_labels.to_csv("row_labels.csv", index=False)

col_label_decomposition = pd.DataFrame(np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)))
col_label_decomposition.to_csv("col_label_decomposition.csv", index=False)



# np.savetxt("to_country_and_edition_at_ixs.txt", 
#            np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), 
#            fmt="%s")

# constructed_data = np.zeros((len(from_countries), len(to_country_and_edition_pairs)))

# target_col_ix = 0
# for edition, data_np in acc_edition2data_np.items():
#     for row in data_np:
        
#         from_country = row[from_country_col_ix]
#         to_country = row[to_country_col_ix]

#         target_row_ix = from_countries.index(from_country)
#         constructed_data[target_row_ix, target_col_ix] = row[points_ix]

#         #THIS DOESNT WORK
#         target_col_ix += 1

# constructed_df = pd.DataFrame(constructed_data, index=from_countries, columns=to_country_and_edition_pairs)
# constructed_df.to_csv("preprocessed.csv")

