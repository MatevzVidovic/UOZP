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
from sklearn.model_selection import train_test_split, cross_val_score
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
    # df["BoW"] = (df["title"] + " ")  + df["authors"].apply(lambda x: " ".join(x)) + " " + df["lead"] + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["subtopics"].apply(lambda x: " ".join(x)) + " " + df["paragraphs"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "" if np.isnan(x) else x)
    df["BoW"] = df["title"] + " " + df["lead"] + " " + df["keywords"].apply(lambda x: " ".join(x)) + " " + df["subtopics"].apply(lambda x: " ".join(x)) + " " + df["gpt_keywords"].apply(lambda x: " ".join(x) if isinstance(x, list) else "" if np.isnan(x) else x)
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
def tfidf(df, topic=None):
    # Initialize vectorizer
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(df["BoW"])
    
    # Get only top Pearson correlation between n_comments and tfidf
    # pearson = np.array([np.corrcoef(tfidf_matrix[:, i].toarray().flatten(), df["n_comments"].to_numpy())[0, 1] for i in range(tfidf_matrix.shape[1])])

    # Picke pearson
    # with open('pearson.pkl', 'wb') as f:
        # pickle.dump(pearson, f)
    # Unplickle pearson

    if topic is None:
        with open('pearson.pkl', 'rb') as f:
            pearson = pickle.load(f)
        pearson = np.abs(pearson)
    elif topic == "topic_sport":
        with open('pearson_sport.pkl', 'rb') as f:
            pearson = pickle.load(f)
        pearson = np.abs(pearson)
    elif topic == "topic_svet":
        with open('pearson_svet.pkl', 'rb') as f:
            pearson = pickle.load(f)
        pearson = np.abs(pearson)

    '''
    top100 = np.argsort(np.abs(pearson))[-100:]
    # Plot all pearson values and top 100 words
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(pearson))
    plt.xlabel('Word index')
    plt.ylabel('Pearson correlation')
    plt.title('Pearson correlation between words and n_comments')
    plt.show()

    words = np.array(vectorizer.get_feature_names_out())[top100]
    # Plot top 100 words
    plt.figure(figsize=(10, 6))
    plt.plot(pearson[top100])
    plt.xticks(range(len(pearson[top100])), words, rotation='vertical')
    plt.xlabel('Word')
    plt.ylabel('Pearson correlation')
    plt.title('Top 100 words and their Pearson correlation')
    plt.show()
    '''

    # '''
    # Get the indices of the top 100 words
    top250 = np.argsort(pearson)[-250:]
    # Remove numbers
    top250 = [i for i in top250 if not any(c.isdigit() for c in vectorizer.get_feature_names_out()[i])]
    # Get the words
    words = np.array(vectorizer.get_feature_names_out())[top250]
    # Get tfidf_matrix with only top 250 words
    tfidf_matrix = tfidf_matrix[:, top250]

    '''
    # Get only words with pearson higher than 0.1
    words = np.array(vectorizer.get_feature_names_out())[np.where(pearson > 0.1)]
    # Get tfidf_matrix with only words with pearson higher than 0.1
    tfidf_matrix = tfidf_matrix[:, np.where(pearson > 0.1)[0]]

    print(f"Number of words: {len(words)}")
    '''

    # Drop BoW column
    df = df.drop(columns=["BoW"])

    return tfidf_matrix, df, words

# Prepare data for training
def prepare_data(df, tfidf_matrix):
    # Target variable
    y = df["n_comments"].to_numpy(dtype=np.float64)

    # Make comparison grpahs
    # plt.subplot(1,2,1)
    # plt.plot(y)
    # plt.xlabel('Article index')
    # plt.ylabel('n_comments')
    # plt.title('Count of articles and n_comments')

    # Lower outliers
    y = np.where(y > 90, 90, y)

    # Lower outliers with upper quartile
    # upper_quartile = df['n_comments'].quantile(0.75)
    # y = np.where(y > upper_quartile.mean(), upper_quartile.mean(), y)

    # Make right graph
    # plt.subplot(1,2,2)
    # plt.plot(y)
    # plt.xlabel('Article index')
    # plt.ylabel('n_comments')
    # plt.title('Count of articles and n_comments')
    # plt.show()

    # Combine tf-idf matrix with other features
    X = hstack([df.drop(columns=["n_comments"]).to_numpy(dtype=np.float64), tfidf_matrix])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Normalize training data
    normalizer = sklearn.preprocessing.Normalizer().fit(X_train)
    X_train = normalizer.transform(X_train)

    return X_train, X_test, y_train, y_test, normalizer

# Get most important features
def most_important_features(model, df, words):
     # Get most important features
    feature_importance = model.coef_
    indices = np.argsort(np.abs(feature_importance))[::-1]
    feature_names = np.array(df.drop(columns=["n_comments"]).columns.to_list() + list(words))
    # Get most important features
    most_important_features = feature_names[indices]
    # Get most important feature values
    most_important_values = feature_importance[indices]
    # Print most important features
    print("\nMost important features:")
    for i in range(15):
        print(f"{most_important_features[i]}: {most_important_values[i]}")

# Find best alpha
def best_alpha(X_train, X_test, y_train, y_test, model, min_alpha, max_alpha, step):
    alphas = np.linspace(min_alpha, max_alpha, step)

    scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        score_r2 = model.score(X_test, y_test)
        score_spearman = np.corrcoef(model.predict(X_test), y_test)[0, 1]
        scores.append((alpha, score_r2, score_spearman))

    # Best alpha
    best_alpha = max(scores, key=lambda x: x[1])[0]
    print(f"Best alpha r2: {best_alpha}")
    best_alpha = max(scores, key=lambda x: x[2])[0]
    print(f"Best alpha spearman: {best_alpha}")

    # Plot scores
    plt.plot(alphas, [score[1] for score in scores], label="R^2")
    plt.plot(alphas, [score[2] for score in scores], label="Spearman's rank correlation coefficient")
    plt.xlabel("Alpha")
    plt.ylabel("Score")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    # Preprocess data
    data, df = preprocess_data("rtvslo_train.json.gzip")

    # Make tf-idf matrix
    tfidf_matrix, df, words = tfidf(df)

    # Prepare data
    X_train, X_test, y_train, y_test, normalizer = prepare_data(df, tfidf_matrix)

    # Fit model
    model = LinearRegression()
    # model = Ridge(alpha=0.01)
    # model = Lasso(alpha=0.01)
    model.fit(X_train, y_train)

    # Predict
    X_test = normalizer.transform(X_test)
    predictions = model.predict(X_test)

    # Most important features
    # most_important_features(model, df, words)


    # Get seperate models for topic_sport and topic_svet
    _, df_sport_svet = preprocess_data("rtvslo_train.json.gzip")

    df_sport = df_sport_svet[df_sport_svet["topic_sport"] == 1]
    df_svet = df_sport_svet[df_sport_svet["topic_svet"] == 1]

    # Make tf-idf matrix
    tfidf_matrix_sport, df_sport, words_sport = tfidf(df_sport, "topic_sport")
    tfidf_matrix_svet, df_svet, words_svet = tfidf(df_svet, "topic_svet")

    # Train model for each topic
    X_train_sport, X_test_sport, y_train_sport, y_test_sport, normalizer_sport = prepare_data(df_sport, tfidf_matrix_sport)
    model_sport = LinearRegression()
    model_sport.fit(X_train_sport, y_train_sport)

    X_train_svet, X_test_svet, y_train_svet, y_test_svet, normalizer_svet = prepare_data(df_svet, tfidf_matrix_svet)
    model_svet = LinearRegression()
    model_svet.fit(X_train_svet, y_train_svet)
    
    # Predict
    X_test_sport = normalizer_sport.transform(X_test_sport)
    predictions_sport = model_sport.predict(X_test_sport)
    X_test_svet = normalizer_svet.transform(X_test_svet)
    predictions_svet = model_svet.predict(X_test_svet)

    # Switch predictions for topic_sport and topic_svet in predictions
    i_sport = 0
    i_svet = 0
    for i in range(len(predictions)):
        if df_sport_svet.iloc[i]["topic_sport"] == 1:
            predictions[i] = predictions_sport[i_sport]
            i_sport += 1
        if df_sport_svet.iloc[i]["topic_svet"] == 1:
            predictions[i] = predictions_svet[i_svet]
            i_svet += 1



    # Evaluate
    print("\nEvaluation for all data:")
    print(f"R^2: {model.score(X_test, y_test):.3f}")
    print(f"Spearman's rank correlation coefficient: {np.corrcoef(predictions, y_test)[0, 1]:.3f}")

    # Evaluate for each topic seperately
    topics = [col for col in df.columns if 'topic_' in col]
    # Merge topics with less than 500 articles
    topics_other = [topic for topic in topics if df[topic].sum() < 1000]
    df["topic_other"] = df[topics_other].sum(axis=1)
    topics = [topic for topic in topics if topic not in topics_other] + ["topic_other"]

    indices_X = {}
    indices_y = {}
    r2 = {}
    spearman = {}
    for topic in topics:
        # Get indeces of topic in X_test
        indices_X[topic] = [i for i in range(X_test.shape[0]) if df.iloc[i][topic] == 1]
        indices_y[topic] = [i for i in range(y_test.shape[0]) if df.iloc[i][topic] == 1]
        # Get X and y
        X = X_test[indices_X[topic]]
        y = y_test[indices_y[topic]]
        # Get predictions from already fitted model
        predictions_topic = predictions[indices_y[topic]]
        # Evaluate
        r2[topic] = model.score(X, y)
        spearman[topic] = np.corrcoef(predictions_topic, y)[0, 1]
        print(f"\nTOPIC: {topic.split('_')[1]}")
        print(f"R^2: {r2[topic]:.3f}")
        print(f"Spearman: {spearman[topic]:.3f}")


    # Get weighted average r2 and spearman
    r2_sum = 0
    spearman_sum = 0
    for topic in topics:
        r2_sum += r2[topic] * len(indices_X[topic])
        spearman_sum += spearman[topic] * len(indices_X[topic])
    print(f"\n\nWeighted average R^2: {r2_sum / len(y_test):.3f}")
    print(f"Weighted average Spearman: {spearman_sum / len(y_test):.3f}")
