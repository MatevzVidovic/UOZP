import json
import gzip
import numpy as np
import pandas as pd
import pickle
import lemmagen3
import matplotlib.pyplot as plt
import string
import sklearn.preprocessing
import sklearn.feature_extraction
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Slovene stopwords
slovene_stopwords = ["a",
                     "ali","april","avgust","b","bi","bil","bila","bile","bili","bilo","biti","blizu","bo","bodo","bojo","bolj","bom","bomo","boste","bova","boš","brez","c","cel","cela","celi","celo","d","da","daleč","dan","danes","datum","december","deset","deseta","deseti","deseto","devet","deveta","deveti","deveto","do","dober","dobra","dobri","dobro","dokler","dol","dolg","dolga","dolgi","dovolj","drug","druga","drugi","drugo","dva","dve","e","eden","en","ena","ene","eni","enkrat","eno","etc.","f","februar","g","g.","ga","ga.","gor","gospa","gospod","h","halo","i","idr.","ii","iii","in","iv","ix","iz","j","januar","jaz","je","ji","jih","jim","jo","julij","junij","jutri","k","kadarkoli","kaj","kajti","kako","kakor","kamor","kamorkoli","kar","karkoli","katerikoli","kdaj","kdo","kdorkoli","ker","ki","kje","kjer","kjerkoli","ko","koder","koderkoli","koga","komu","kot","kratek","kratka","kratke","kratki","l","lahka","lahke","lahki","lahko","le","lep","lepa","lepe","lepi","lepo","leto","m","maj","majhen","majhna","majhni","malce","malo","manj","marec","me","med","medtem","mene","mesec","mi","midva","midve","mnogo","moj","moja","moje","mora","morajo","moram","moramo","morate","moraš","morem","mu","n","na","nad","naj","najina","najino","najmanj","naju","največ","nam","narobe","nas","nato","nazaj","naš","naša","naše","ne","nedavno","nedelja","nek","neka","nekaj","nekatere","nekateri","nekatero","nekdo","neke","nekega","neki","nekje","neko","nekoga","nekoč","ni","nikamor","nikdar","nikjer","nikoli","nič","nje","njega","njegov","njegova","njegovo","njej","njemu","njen","njena","njeno","nji","njih","njihov","njihova","njihovo","njiju","njim","njo","njun","njuna","njuno","no","nocoj","november","npr.","o","ob","oba","obe","oboje","od","odprt","odprta","odprti","okoli","oktober","on","onadva","one","oni","onidve","osem","osma","osmi","osmo","oz.","p","pa","pet","peta","petek","peti","peto","po","pod","pogosto","poleg","poln","polna","polni","polno","ponavadi","ponedeljek","ponovno","potem","povsod","pozdravljen","pozdravljeni","prav","prava","prave","pravi","pravo","prazen","prazna","prazno","prbl.","precej","pred","prej","preko","pri","pribl.","približno","primer","pripravljen","pripravljena","pripravljeni","proti","prva","prvi","prvo","r","ravno","redko","res","reč","s","saj","sam","sama","same","sami","samo","se","sebe","sebi","sedaj","sedem","sedma","sedmi","sedmo","sem","september","seveda","si","sicer","skoraj","skozi","slab","smo","so","sobota","spet","sreda","srednja","srednji","sta","ste","stran","stvar","sva","t","ta","tak","taka","take","taki","tako","takoj","tam","te","tebe","tebi","tega","težak","težka","težki","težko","ti","tista","tiste","tisti","tisto","tj.","tja","to","toda","torek","tretja","tretje","tretji","tri","tu","tudi","tukaj","tvoj","tvoja","tvoje","u","v","vaju","vam","vas","vaš","vaša","vaše","ve","vedno","velik","velika","veliki","veliko","vendar","ves","več","vi","vidva","vii","viii","visok","visoka","visoke","visoki","vsa","vsaj","vsak","vsaka","vsakdo","vsake","vsaki","vsakomur","vse","vsega","vsi","vso","včasih","včeraj","x","z","za","zadaj","zadnji","zakaj","zaprta","zaprti","zaprto","zdaj","zelo","zunaj","č","če","često","četrta","četrtek","četrti","četrto","čez","čigav","š","šest","šesta","šesti","šesto","štiri","ž","že"]


# Preprocess data
def preprocess_data(path):
    # open rtvslo_train.json.gzip
    with gzip.open(path, "rt") as file:
        data = json.load(file)

    # Convert to pandas dataframe
    df = pd.DataFrame(data)


    # One-hot encode topics
    df = pd.concat([df, pd.get_dummies(df["topics"], prefix="topic")], axis=1)
    # Get subtopics from url
    df["subtopics"] = df["url"].apply(lambda x: x.split("/")[4:-2])
    # One-hot encode each subtopic
    # df = pd.concat([df, pd.get_dummies(df["subtopics"].explode(), prefix="subtopic")], axis=1)


    # Average number of comments per "topic"
    # print(f"Average number of comments per topic: {df.groupby('topics')['n_comments'].mean()}")
    # Plot number of comments for each "topic"
    # df.groupby('topics')['n_comments'].mean().plot(kind='bar')
    # Number of unique subtopics
    # print(f"Number of unique subtopics: {df['subtopics'].explode().nunique()}")
    # Number of unique subtopics per "topic"
    # print(f"Number of unique subtopics per topic:\n{df['subtopics'].explode().groupby(df['topics']).nunique()}")
    # Number of unique authors 
    # print(f"Number of unique authors: {df['authors'].explode().nunique()}")
    # Number of unique authors per "topic"
    # print(f"Number of unique authors per topic: {df['authors'].explode().groupby(df['topics']).nunique()}")
    # Number of articles per "topic"
    # print(f"Number of articles per topic: {df['topics'].value_counts()}")


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
    df["BoW"] = df["title"] + " " + df["authors"].apply(lambda x: " ".join(x)) + " " + df["lead"] + " " + df["paragraphs"].apply(lambda x: " ".join(x)) + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else x) + " " + df["subtopics"].apply(lambda x: " ".join(x))
    # df["BoW"] = df["title"] + " " + df["authors"].apply(lambda x: " ".join(x)) + " " + df["lead"] + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x)) + " " + df["subtopics"].apply(lambda x: " ".join(x))
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

    return data, df

# Make tf-idf matrix
def tfidf(df):
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

    return tfidf_matrix, df, vectorizer

# Prepare data for training
def prepare_data(df, tfidf_matrix):
    # Target variable
    y = df["n_comments"].to_numpy(dtype=np.float64)

    # Combine tf-idf matrix with other features
    X = hstack([tfidf_matrix, df.drop(columns=["n_comments"]).to_numpy(dtype=np.float64)])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Normalize training data
    normalizer = sklearn.preprocessing.Normalizer().fit(X_train)
    X_train = normalizer.transform(X_train)

    return X_train, X_test, y_train, y_test, normalizer


# Preprocess data: Bag of words --> lemmatization --> (filter --> clustering) --> tfidf --> (filter) --> (pca) -->  regression
data, df = preprocess_data("rtvslo_train.json.gzip")

'''
# Pickle data and df
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)
with open("df_BoW.pkl", "wb") as file:
    pickle.dump(df, file)
'' '
# Unpickle data and df
with open("data.pkl", "rb") as file:
    data = pickle.load(file)
with open("df_BoW.pkl", "rb") as file:
    df = pickle.load(file)
# '''

tfidf_matrix, df, vectorizer = tfidf(df)

'''
# Pickle tfidf_matrix, df, vectorizer
with open("tfidf_matrix.pkl", "wb") as file:
    pickle.dump(tfidf_matrix, file)
with open("df.pkl", "wb") as file:
    pickle.dump(df, file)
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
# '' '
# Unpickle tfidf_matrix and df
with open("tfidf_matrix.pkl", "rb") as file:
    tfidf_matrix = pickle.load(file)
with open("df.pkl", "rb") as file:
    df = pickle.load(file)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
# '''

# Prepare data
X_train, X_test, y_train, y_test, normalizer = prepare_data(df, tfidf_matrix)

'''
# Pickle X_train, X_test, y_train, y_test
with open("X_train.pkl", "wb") as file:
    pickle.dump(X_train, file)
with open("X_test.pkl", "wb") as file:
    pickle.dump(X_test, file)
with open("y_train.pkl", "wb") as file:
    pickle.dump(y_train, file)
with open("y_test.pkl", "wb") as file:
    pickle.dump(y_test, file)
with open("normalizer.pkl", "wb") as file:
    pickle.dump(normalizer, file)
# '' '
# Unpickle X_train, X_test, y_train, y_test
with open("X_train.pkl", "rb") as file:
    X_train = pickle.load(file)
with open("X_test.pkl", "rb") as file:
    X_test = pickle.load(file)
with open("y_train.pkl", "rb") as file:
    y_train = pickle.load(file)
with open("y_test.pkl", "rb") as file:
    y_test = pickle.load(file)
with open("normalizer.pkl", "rb") as file:
    normalizer = pickle.load(file)
# '''


# Fit model
# model = LinearRegression()
model = Ridge(alpha=0.3)
# model = Lasso(alpha=0.5)
model.fit(X_train, y_train)


# Predict
X_test = normalizer.transform(X_test)
predictions = model.predict(X_test)


# Evaluate
print(f"R^2: {model.score(X_test, y_test):.3f}") # 0.1 or 0.2 R^2 is already good
print(f"Spearman's rank correlation coefficient: {np.corrcoef(predictions, y_test)[0, 1]:.3f}")
print(f"RMSE: {np.sqrt(np.mean((predictions - y_test) ** 2)):.3f}")




# Get most important features
feature_importance = model.coef_
# Get indices of most important features
indices = np.argsort(np.abs(feature_importance))[::-1]
# Get feature names
feature_names = np.array(list(vectorizer.get_feature_names_out()) + df.drop(columns=["n_comments"]).columns.to_list())
# Get most important features
most_important_features = feature_names[indices]
# Get most important feature values
most_important_values = feature_importance[indices]
# Print most important features
print("\nMost important features:")
for i in range(15):
    print(f"{most_important_features[i]}: {most_important_values[i]}")


# Print a part of predictions vs actual values
# print("\nPredictions vs actual values:")
# for i in range(50):
#     print(f"{predictions[i]:.2f} . . . {y_test[i]}")
