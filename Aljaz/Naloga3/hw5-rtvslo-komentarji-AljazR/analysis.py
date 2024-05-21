import pickle
import matplotlib.pyplot as plt
import numpy as np

def test1():
    
    # unpicke tfidf_matrix
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    # calculate the sum of each column in the TF-IDF matrix and sort them in ascending order
    sums = tfidf_matrix.sum(axis=0)[0]
    # change from matrix to array
    sums = sums.A1
    # sort the sums
    sums.sort()


    # With matplotlib, plot distribution of the sum of each column in the TF-IDF matrix, don'r use all the words, just every 100th word
    plt.figure(figsize=(10, 6))
    plt.plot(sums)
    plt.xlabel('Word index')
    plt.ylabel('Sum')
    plt.title('Distribution of the sum of each column in the TF-IDF matrix')
    plt.show()



    # average number of n_comments in year 2023 and year 2024
    # unpicke df
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)

    # filter the data for years 2023 and 2024
    df_2023 = df[df['year'] == 2023]
    df_2024 = df[df['year'] == 2024]

    # Average of upper quartile of n_comments in year 2023 and year 2024
    upper_quartile_2023 = df_2023['n_comments'].quantile(0.75)
    upper_quartile_2024 = df_2024['n_comments'].quantile(0.75)

    # Median of 2023 and 2024
    median_2023 = df_2023['n_comments'].median()
    median_2024 = df_2024['n_comments'].median()

    print(f'Median of n_comments in year 2023: {median_2023}')
    print(f'Median of n_comments in year 2024: {median_2024}')

    print(f'Upper quartile of n_comments in year 2023: {upper_quartile_2023.mean()}')
    print(f'Upper quartile of n_comments in year 2024: {upper_quartile_2024.mean()}')

    # calculate the average number of n_comments in year 2023 and year 2024
    average_2023 = df_2023['n_comments'].mean()
    average_2024 = df_2024['n_comments'].mean()

    print(f'Average number of n_comments in year 2023: {average_2023}')
    print(f'Average number of n_comments in year 2024: {average_2024}')

    # Make a graph with y-axis being count of articles and x-axis being n_comments, plot it, don't use histograms
    plt.figure(figsize=(10, 6))
    plt.plot(df_2023['n_comments'], label='2023')
    plt.plot(df_2024['n_comments'], label='2024')
    plt.xlabel('Article index')
    plt.ylabel('n_comments')
    plt.title('Count of articles and n_comments')
    plt.legend()
    plt.show()

    # Print average n_comments
    print(f'Average n_comments in year 2023: {average_2023}')

    # Graph of how many articles have n_comments in the range of 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100, more than 100
    # filter the data for n_comments in the range of 0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100, more than 100
    df_2023_0_10 = df[(df['n_comments'] >= 0) & (df['n_comments'] <= 10)]
    df_2023_10_20 = df[(df['n_comments'] > 10) & (df['n_comments'] <= 20)]
    df_2023_20_30 = df[(df['n_comments'] > 20) & (df['n_comments'] <= 30)]
    df_2023_30_40 = df[(df['n_comments'] > 30) & (df['n_comments'] <= 40)]
    df_2023_40_50 = df[(df['n_comments'] > 40) & (df['n_comments'] <= 50)]
    df_2023_50_60 = df[(df['n_comments'] > 50) & (df['n_comments'] <= 60)]
    df_2023_60_70 = df[(df['n_comments'] > 60) & (df['n_comments'] <= 70)]
    df_2023_70_80 = df[(df['n_comments'] > 70) & (df['n_comments'] <= 80)]
    df_2023_80_90 = df[(df['n_comments'] > 80) & (df['n_comments'] <= 90)]
    df_2023_90_100 = df[(df['n_comments'] > 90) & (df['n_comments'] <= 100)]
    df_2023_more_100 = df[df['n_comments'] > 100]

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', 'more than 100'], [len(df_2023_0_10), len(df_2023_10_20), len(df_2023_20_30), len(df_2023_30_40), len(df_2023_40_50), len(df_2023_50_60), len(df_2023_60_70), len(df_2023_70_80), len(df_2023_80_90), len(df_2023_90_100), len(df_2023_more_100)])
    plt.xlabel('n_comments')
    plt.ylabel('Count of articles')
    plt.title('Count of articles and n_comments')
    plt.show()

# with open('tfidf_matrix.pkl', 'rb') as f:
#     tfidf_matrix = pickle.load(f)
# with open('df.pkl', 'rb') as f:
#     df = pickle.load(f)
# with open('vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)


def pearson1():

    print(tfidf_matrix.shape)
    # pearson = np.array([np.corrcoef(tfidf_matrix[:, i].toarray().flatten(), df["n_comments"].to_numpy())[0, 1] for i in range(tfidf_matrix.shape[1])])

    # picke pearson
    # with open('pearson.pkl', 'wb') as f:
        # pickle.dump(pearson, f)

    # unplickle pearson
    with open('pearson.pkl', 'rb') as f:
        pearson = pickle.load(f)

    # Get abs
    pearson = np.abs(pearson)

    # Get the indices of the top 100 words
    top100 = np.argsort(pearson)[-250:]

    # remove numbers
    top100 = [i for i in top100 if not any(c.isdigit() for c in vectorizer.get_feature_names_out()[i])]

    # Get the words
    words = np.array(vectorizer.get_feature_names_out())[top100]

    # Plot all pearson values and top 100 words
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.sort(pearson))
    # plt.xlabel('Word index')
    # plt.ylabel('Pearson correlation')
    # plt.title('Pearson correlation between words and n_comments')
    # plt.show()

    # Plot top 100 words
    plt.figure(figsize=(10, 6))
    plt.plot(pearson[top100])
    plt.xticks(range(len(pearson[top100])), words, rotation='vertical')
    plt.xlabel('Word')
    plt.ylabel('Pearson correlation')
    plt.title('Top 100 words and their Pearson correlation')
    plt.show()

# number of articles per topic
def test2(df):

    # number of articles per topic, which are one-hot encoded as topics_*
    # get the columns of topics
    topics = [col for col in df.columns if 'topic_' in col]

    # get the number of articles per topic
    count = {}
    for topic in topics:
        # save number of rows
        count[topic] = len(df[topic].where(df[topic] == 1).dropna())

    # get a number of rows in pandas dataframe

        

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(topics, count.values())
    plt.xlabel('Topic')
    plt.ylabel('Number of articles')
    plt.title('Number of articles per topic')
    plt.show()

# test2(df)

import json
import gzip

with gzip.open("rtvslo_train.json.gzip", "rt") as file:
        data = json.load(file)

# split into train and test data
train_data = data[:int(0.7 * len(data))]
test_data = data[int(0.7 * len(data)):]

# save train and test data as gzib json
with gzip.open("train_data.json.gz", "wt") as file:
    json.dump(train_data, file)

with gzip.open("test_data.json.gz", "wt") as file:
    json.dump(test_data, file)