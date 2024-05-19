import argparse
import json
import gzip
import os
import numpy as np

import pandas as pd
import lemmagen3
import string
import sklearn.preprocessing
import sklearn.feature_extraction
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


# Slovene stopwords
slovene_stopwords = ["a",
                     "ali","april","avgust","b","bi","bil","bila","bile","bili","bilo","biti","blizu","bo","bodo","bojo","bolj","bom","bomo","boste","bova","boš","brez","c","cel","cela","celi","celo","d","da","daleč","dan","danes","datum","december","deset","deseta","deseti","deseto","devet","deveta","deveti","deveto","do","dober","dobra","dobri","dobro","dokler","dol","dolg","dolga","dolgi","dovolj","drug","druga","drugi","drugo","dva","dve","e","eden","en","ena","ene","eni","enkrat","eno","etc.","f","februar","g","g.","ga","ga.","gor","gospa","gospod","h","halo","i","idr.","ii","iii","in","iv","ix","iz","j","januar","jaz","je","ji","jih","jim","jo","julij","junij","jutri","k","kadarkoli","kaj","kajti","kako","kakor","kamor","kamorkoli","kar","karkoli","katerikoli","kdaj","kdo","kdorkoli","ker","ki","kje","kjer","kjerkoli","ko","koder","koderkoli","koga","komu","kot","kratek","kratka","kratke","kratki","l","lahka","lahke","lahki","lahko","le","lep","lepa","lepe","lepi","lepo","leto","m","maj","majhen","majhna","majhni","malce","malo","manj","marec","me","med","medtem","mene","mesec","mi","midva","midve","mnogo","moj","moja","moje","mora","morajo","moram","moramo","morate","moraš","morem","mu","n","na","nad","naj","najina","najino","najmanj","naju","največ","nam","narobe","nas","nato","nazaj","naš","naša","naše","ne","nedavno","nedelja","nek","neka","nekaj","nekatere","nekateri","nekatero","nekdo","neke","nekega","neki","nekje","neko","nekoga","nekoč","ni","nikamor","nikdar","nikjer","nikoli","nič","nje","njega","njegov","njegova","njegovo","njej","njemu","njen","njena","njeno","nji","njih","njihov","njihova","njihovo","njiju","njim","njo","njun","njuna","njuno","no","nocoj","november","npr.","o","ob","oba","obe","oboje","od","odprt","odprta","odprti","okoli","oktober","on","onadva","one","oni","onidve","osem","osma","osmi","osmo","oz.","p","pa","pet","peta","petek","peti","peto","po","pod","pogosto","poleg","poln","polna","polni","polno","ponavadi","ponedeljek","ponovno","potem","povsod","pozdravljen","pozdravljeni","prav","prava","prave","pravi","pravo","prazen","prazna","prazno","prbl.","precej","pred","prej","preko","pri","pribl.","približno","primer","pripravljen","pripravljena","pripravljeni","proti","prva","prvi","prvo","r","ravno","redko","res","reč","s","saj","sam","sama","same","sami","samo","se","sebe","sebi","sedaj","sedem","sedma","sedmi","sedmo","sem","september","seveda","si","sicer","skoraj","skozi","slab","smo","so","sobota","spet","sreda","srednja","srednji","sta","ste","stran","stvar","sva","t","ta","tak","taka","take","taki","tako","takoj","tam","te","tebe","tebi","tega","težak","težka","težki","težko","ti","tista","tiste","tisti","tisto","tj.","tja","to","toda","torek","tretja","tretje","tretji","tri","tu","tudi","tukaj","tvoj","tvoja","tvoje","u","v","vaju","vam","vas","vaš","vaša","vaše","ve","vedno","velik","velika","veliki","veliko","vendar","ves","več","vi","vidva","vii","viii","visok","visoka","visoke","visoki","vsa","vsaj","vsak","vsaka","vsakdo","vsake","vsaki","vsakomur","vse","vsega","vsi","vso","včasih","včeraj","x","z","za","zadaj","zadnji","zakaj","zaprta","zaprti","zaprto","zdaj","zelo","zunaj","č","če","često","četrta","četrtek","četrti","četrto","čez","čigav","š","šest","šesta","šesti","šesto","štiri","ž","že"]

class RTVSlo:

    def __init__(self):
        self.normalizer = None
        self.model = None

    def preprocess(self, data: list):
        # Convert to pandas dataframe
        df = pd.DataFrame(data)

        # One-hot encode topics
        df = pd.concat([df, pd.get_dummies(df["topics"], prefix="topic")], axis=1)
        # Get subtopics from url
        df["subtopics"] = df["url"].apply(lambda x: x.split("/")[4:-2])
        # One-hot encode each subtopic
        # df = pd.concat([df, pd.get_dummies(df["subtopics"].explode(), prefix="subtopic")], axis=1)

        # Number of paragraphs
        df["n_paragraphs"] = df["paragraphs"].apply(lambda x: len(x))
        # Length of paragraphs
        df["length"] = df["paragraphs"].apply(lambda x: len(" ".join(x)))
        # Save length of figures array in new column n_pictures
        df["n_pictures"] = df["figures"].apply(lambda x: len(x))


        # Convert to datetime
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%dT%H:%M:%S")
        # Make a column for years
        df["year"] = df["date"].dt.year
        # One-hot encode months
        df = pd.concat([df, pd.get_dummies(df["date"].dt.month, prefix="month")], axis=1)
        # One-hot encode days
        df = pd.concat([df, pd.get_dummies(df["date"].dt.dayofweek, prefix="day")], axis=1)
        # One-hot encode hours
        df = pd.concat([df, pd.get_dummies(df["date"].dt.hour, prefix="hour")], axis=1)


        # Make bag of words
        df["BoW"] = df["title"] + " " + df["authors"].apply(lambda x: " ".join(x)) + " " + df["lead"] + " " + df["paragraphs"].apply(lambda x: " ".join(x)) + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["subtopics"].apply(lambda x: " ".join(x))
        # Add gpt_keywords to BoW
        df["BoW"] = df["BoW"] + " " + df["gpt_keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "" if np.isnan(x) else x)
        # Lowercase
        df["BoW"] = df["BoW"].apply(lambda x: x.lower())
        # Removes all punctuation
        df["BoW"] = df["BoW"].str.translate(str.maketrans("", "", string.punctuation))
        # Lemmatize
        lemmatizer = lemmagen3.Lemmatizer('sl')
        df["BoW"] = df["BoW"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
        # Remove stopwords using slovene_stopwords
        df["BoW"] = df["BoW"].apply(lambda x: " ".join([word for word in x.split() if word not in slovene_stopwords]))


        # Drop unnecessary columns
        df = df.drop(columns=["url", "category", "figures", "id", "date", "topics", "subtopics", "title", "authors", "lead", "paragraphs", "keywords", "gpt_keywords"])
        

        # Initialize vectorizer
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(df["BoW"])

        # Filter tf-idf based on the sum of each column 
        # The sum of a column in the TF-IDF matrix captures the overall relevance of a word, accounting for both its local occurrence and its global importance
        # avg_scores = np.mean(np.sum(tfidf_matrix, axis=0))
        # tfidf_matrix = tfidf_matrix[:, np.where(np.sum(tfidf_matrix, axis=0) > avg_scores / 2)[0]]
        
        # Drop BoW column
        df = df.drop(columns=["BoW"])

        return df, tfidf_matrix

    def fit(self, train_data: list):
        # Preprocess data
        df, tfidf_matrix = self.preprocess(train_data)

        # Target variable
        y = df["n_comments"].to_numpy(dtype=np.float64)

        # Combine tf-idf matrix with other features
        X = hstack([tfidf_matrix, df.drop(columns=["n_comments"]).to_numpy(dtype=np.float64)])

        # Normalize training data
        normalizer = sklearn.preprocessing.Normalizer().fit(X)
        X = normalizer.transform(X)
        self.normalizer = normalizer

        # Fit model
        # self.model = LinearRegression()
        self.model = Ridge(alpha=0.3)
        # self.model = Lasso(alpha=0.5)
        self.model.fit(X, y)

    def predict(self, test_data: list) -> np.array:

        df, tfidf_matrix = self.preprocess(test_data)

        X = hstack([tfidf_matrix, df.drop(columns=["n_comments"]).to_numpy(dtype=np.float64)])
        X = self.normalizer.transform(X)
        return self.model.predict(X)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = read_json(args.train_data_path)
    test_data = read_json(args.test_data_path)

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
