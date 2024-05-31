import argparse
import json
import gzip
import os
import numpy as np
import pandas as pd
import lemmagen3
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def cross_validation(data, k=5):
    mae_arr = []
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        target_variable = np.array([article['n_comments'] for article in test_data])
        
        rtv = RTVSlo()
        rtv.fit(train_data)
        predictions = rtv.predict(test_data)
        
        mae_arr.append(np.mean(np.abs(predictions - target_variable)))

    print(f'MAE: {np.mean(mae_arr):.2f}')

class RTVSlo:

    def __init__(self):
        self.vectorizer = None
        self.standarizer = None
        self.features = None
        self.model = None
    
    def preprocess(self, data, train=True):
        # Create a DataFrame
        df = pd.DataFrame(data)

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%dT%H:%M:%S")
        # One-hot encode day of the week
        df = pd.concat([df, pd.get_dummies(df["date"].dt.dayofweek, prefix="day")], axis=1)
        # One-hot encode hours
        df = pd.concat([df, pd.get_dummies(df["date"].dt.hour, prefix="hour")], axis=1)

        # One-hot encode authors
        all_authors = set()
        for authors in df["authors"]:
            all_authors.update(authors)
        df_authors = pd.DataFrame(0, columns=list(all_authors), index=range(df.shape[0]))
        for i, authors in enumerate(df["authors"]):
            for author in authors:
                df_authors.at[i, author] = 1      
        df = pd.concat([df, df_authors], axis=1)

        # One-hot encode topics
        df = pd.concat([df, pd.get_dummies(df["topics"], prefix="topic")], axis=1)

        # Get subtopics from url
        df["subtopics"] = df["url"].apply(lambda x: x.split("/")[4:-2])
        # One-hot encode subtopics
        all_subtopics = set()
        for subtopics in df["subtopics"]:
            all_subtopics.update(subtopics)
        df_subtopics = pd.DataFrame(0, columns=list(all_subtopics), index=range(df.shape[0]))
        for i, subtopics in enumerate(df["subtopics"]):
            for subtopic in subtopics:
                df_subtopics.at[i, subtopic] = 1
        df = pd.concat([df, df_subtopics], axis=1)

        # Make bag of words
        df["BoW"] = df["title"] + " " + df["lead"] + " " + df["paragraphs"].apply(lambda x: " ".join(x)) + " " + df["figures"].apply(lambda x: " ".join([figure["caption"] for figure in x if "caption" in figure]))
        df["BoW"] = df["BoW"] + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "" if np.isnan(x) else x)
        # Lowercase
        df["BoW"] = df["BoW"].apply(lambda x: x.lower())
        # Removes all punctuation
        df["BoW"] = df["BoW"].str.translate(str.maketrans("", "", string.punctuation))
        # Lemmatize
        lemmatizer = lemmagen3.Lemmatizer('sl')
        df["BoW"] = df["BoW"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))

        # TF-IDF
        if train:
            # Initialize vectorizer
            vectorizer = TfidfVectorizer()
            # Fit and transform
            vectorizer.fit(df["BoW"])
            tfidf_matrix = vectorizer.transform(df["BoW"])
            self.vectorizer = vectorizer
        else:
            tfidf_matrix = self.vectorizer.transform(df["BoW"])

        # Prepare target variable
        if train:
            y = df["n_comments"].to_numpy(dtype=np.float64)
            y = np.sqrt(y)
            # y[y > 90] = 90

        # Drop unnecessary columns
        drop_columns = ["BoW", "url", "authors", "date",
                        "title", "paragraphs", "figures",
                        "lead", "topics", "keywords",
                        "gpt_keywords", "n_comments",
                        "id", "category", "subtopics"]
        for col in drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Add missing columns or delete additional columns from test data
        if train:
            self.features = df.columns
        else:
            if self.features is not None:
                new_cols = [col for col in self.features if col not in df.columns]
                df_missing_cols = pd.DataFrame(0, columns=new_cols, index=range(df.shape[0]))
                df = pd.concat([df, df_missing_cols], axis=1)
            else:
                df = df.copy()

            if set(self.features) != set(df.columns):
                for col in df.columns:
                    if col not in self.features and col != "n_comments":
                        df = df.drop(columns=[col])

        # Marge TF-IDF matrix with other features
        X = hstack([tfidf_matrix, df.to_numpy(dtype=np.float64)])

        # Standardize X
        if train:
            standarizer = StandardScaler(with_mean=False).fit(X)
            X = standarizer.transform(X)
            self.standarizer = standarizer

            return X, y
        else:
            X = self.standarizer.transform(X)

            return X

    def fit(self, train_data: list):
        # Preprocess data
        X, y = self.preprocess(train_data)

        # Train model
        self.model = Ridge(alpha=1)
        self.model.fit(X, y)
        
    def predict(self, test_data: list) -> np.array:
        # Preprocess data
        X = self.preprocess(test_data, train=False)

        # Predict
        predictions = self.model.predict(X) ** 2

        return predictions


def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = read_json(args.train_data_path)
    test_data = read_json(args.test_data_path)
    '''

    train_data = read_json('data/rtvslo_train.json.gz')
    test_data = read_json('data/rtvslo_test.json.gz')

    # '''
    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)
    '''

    cross_validation(train_data)
    # '''


if __name__ == '__main__':
    main()
