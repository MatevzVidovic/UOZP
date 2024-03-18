

import pandas as pd
import numpy as np

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
if True:
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





from_countries = raw_data["From country"].unique()
to_countries = raw_data["To country"].unique()












    