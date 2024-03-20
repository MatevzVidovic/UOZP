
import numpy as np
import pandas as pd



constructed_data = pd.read_csv("preprocessed.csv").values
# np.savetxt("to_country_and_edition_at_ixs.txt", np.column_stack((to_country_at_corresponding_ix, edition_at_corresponding_ix)), fmt="%s")

col_labels = pd.read_csv("col_labels.csv").values

row_labels = pd.read_csv("row_labels.csv").values

col_label_decomposition = pd.read_csv("col_label_decomposition.csv").values


print(constructed_data)
