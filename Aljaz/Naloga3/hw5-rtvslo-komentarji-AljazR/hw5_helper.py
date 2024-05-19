import json
import gzip
import numpy as np
import pandas as pd
import sklearn
import pickle
import lemmagen3
import matplotlib.pyplot as plt
import string

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
    df["BoW"] = df["title"] + " " + df["authors"].apply(lambda x: " ".join(x)) + " " + df["lead"] + " " + df["paragraphs"].apply(lambda x: " ".join(x)) + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x)) + " " + df["subtopics"].apply(lambda x: " ".join(x))
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
    df = df.drop(columns=["url", "category", "figures", "id", "date", "topics"])

    return data, df

# Make tf-idf matrix
def tfidf(X):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    X = vectorizer.fit_transform(df["BoW"])

    # Plot distribution of X.sum(axis=0)
    plt.hist(X.sum(axis=0), bins=100)

    # Filter tf-idf based on the sum of each column 
    # The sum of a column in the TF-IDF matrix captures the overall relevance of a word, accounting for both its local occurrence and its global importance
    # X = X[:, np.where(X.sum(axis=0) > 50)[0]]


    # Add to dataframe - too large datasets --> use sparse matrices
    # df = pd.concat([df, pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)
    
    # Drop unnecessary columns
    df = df.drop(columns=["subtopics", "title", "authors", "lead", "paragraphs", "keywords", "gpt_keywords", "BoW"])

    return X, df


# data, df = preprocess_data("rtvslo_train.json.gzip")

'''
# Pickle data and df
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)
with open("df.pkl", "wb") as file:
    pickle.dump(df, file)
'''
# Unpickle data and df
with open("data.pkl", "rb") as file:
    data = pickle.load(file)
with open("df.pkl", "rb") as file:
    df = pickle.load(file)
# '''

tfidf_matrix, df = tfidf(df)



# TODO:
# make tfidf
# poglej kako dela test_hw5.py, da bos vedu kaj more bit v hw5.py
'''
Preprocess data:
Bag of words --> lemaztizacija --> (filtriranje --> clustering) --> tfidf --> (filtriranje) --> (pca) -->  regresija

Additional ideas:
Prevedi vse v anglescino in uporabi angleske knjiznice za lemaztizacijo in stopwords
Za vsako kategorjo svoj model?
Bi za naslov uporabu k-mers?
Predpriprava kljucnikov: lahko jih clusteras med sabo, da zmanjsas njihovo stevilo
Keywordi od mmcja niso dobri
Mention je nekonsistenten, ampak js bi ga uporabu (sploh ga ni v podatkih)

Evaluation:
Uporabi Spearmanova rang korelacija in R^2
Tf-idf nad vsem tekstom je en dubu R^2 0.45
0.1 ali 0.2 R^2 je ze vredu
'''