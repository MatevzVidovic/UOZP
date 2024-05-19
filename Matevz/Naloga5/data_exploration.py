
import hw4

"""

UOZP naloga:
- prebrat podatke in jih dat v parseano obliko
- delitev na testno, validacijsko in učno. Pa mogoče kar nek CV izvajat ker pomoje to hočejo.

- izbrane kategorije od spodaj navzgor:
(tfidf := stemming/NLTK za slo. Potem tfidf in rezanje manj pojavljenih/manj pomembnih po feature selectionu.)

- gpt_keywords - tfidf. Teh baje pogosto ni. So za: target_topics = [“gospodarstvo”, “sport”, “kultura”, 
“zabava-in-slog”]. Tako da samo za te modele jih upoštevaj.
- keywords -  tfidf
- !!!topics!!! - preverit koliko jih je. Če jih je več, naredit urejen tuple kot key. Vse te možnosti se
 potem one-hot encode-a. Pa odreže se te, ki nimajo nekega velikega vpliva po feature selectionu, oziroma
   ki se premalo pojavijo. Ne naredim tega, da enega od možnih ne one-ho-encodeaš, da bi bil tvoj baseline
     in je pravilno parametriziran model - ker moje baseline je, da pač ni ujemanja topica s temi, ki so
       one-ho encodani in je to potem baseline.
- lead - tfidf
- naredit number of figures kot svojo značilko in samo dat kot šrtevilko
- ??paragraphs??
- ??title??
- date - dobit ven leto, mesec, in uro. Leto vstavit kot številko in le dobit linearen trend.
 Mesec tudi kot številko ampak ne le linearno (dodat sinusoido, kosinusoido, koren, in kvadrat
   (da bolje ujamem nihanja skozi leto). Ure pa dat v neke buckete po recimo 3 ure in to one-hot encodeat
     - to je pomembno ker od 22.00 ni več komentarjev do jutra in zato to zelo vpliva za pozne članke in tako.
- !!!authors!!! - isto kot topics.
- !!!URL!!! - parseat ker se v njem skriva glavna kategorija in tudi sub-categories.
 Potem za te subcategories isto kot za topics.

- mogoče za vsak topic in (1 kategorija za pomanjkanje topica) naredit svoj model. 
Zna biti zelo uporabno. Lahko model poponoma prilagodi topicu
 (tako da vsak ta model potrebuje veliko manj parametrov pa imamo v celoti višji accuracy.
   In je hitreje za trenirat. Je pa problem, da tako zelo zmanjšaš število training primerov
     ki applyjajo na posamezen model. Zato naredi to samo za večje topice in poskrbi, da je v tej "no topic"
       kategoriji vseeno dosti primerov.

- distribucija števila komentarjev je nabasana proti nižjemu številu in ima potem zelo dolg rep. 
To mi zna zjebat model, ker bojo za kvadratne napake ekstra vlekli v svojo smer. Mogoče odstranit te osamelce.
 Mogoče apllyat logaritem na število komentarjev. Mogoče samo za komentarje nad nekim številom 
 (recimo v 90. percentilu) narediti mapping v: 90. percentil št. kom. + log (pravo število komentarjev). 
 Tako bodo še vedno lepo uporabni za treniranje in jih bo fittalo kot visoko število komentarjev, 
 pa ne bodo uničili treniranja.

- mogoče supervised PCA, ampak se mi zdajle ne da. Pa ne vem če ima to kakšna od dovoljenih knjižnic.

- potem vse te matrike tfidfja in one-hot encodinga, ki sem jih prej naredil, concatat v eno matriko
 in jih dat v linear regression model z L1 regularizacijo.
- z validacijsko množico določit stopnjo regularizacije - kot nek procent R2 ko je regularizacija skoraj nična.

- testirat na testni

- s tem procesom delat 2CV5.
- potem zgradit model z deljenjem samo na train in valid - in to je moj končni model.
 (kot smo za CV govorili na predavanjih)
 """




import gzip
import json

import pickle


"""
if True:
    import pickle

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)


    # Path to save the pickled file
    pickle_file_path = './data/rtvslo_train.pkl'

    # Pickle the data
    with open(pickle_file_path, 'wb') as pf:
        pickle.dump(data, pf)

else:

    # Path to your pickle file
    pickle_file_path = './data/rtvslo_train.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data = pickle.load(pf)
"""




class DataClass:
    def __init__(self, data):
        self.URLs = []
        self.authors = []
        self.dates = []
        self.nums_of_figs = []
        self.leads = []
        self.topics = []
        self.keywords = []
        self.gpt_keywords = []
        self.num_of_comments = []

        self.length = 0
        self.num_of_ommited_by_topic = 0

        self.extract_data(data)

    def extract_data(self, data):
        for article in data:
            try:
                self.topics.append(article['topics'])
            except KeyError:
                print("Article has no topics. Ommiting it. URL:")
                print(article['url'])
                self.num_of_ommited_by_topic += 1
                continue

            
            self.URLs.append(article['url'])
            self.authors.append(article['authors'])
            self.dates.append(article['date'])
            self.nums_of_figs.append(len(article['figures']))
            self.leads.append(article['lead'])
            self.keywords.append(article['keywords'])
            self.gpt_keywords.append(article['gpt_keywords'])
            self.num_of_comments.append(article['n_comments'])

            self.length += 1


if True:

    # Path to your .json.gzip file
    file_path = './data/rtvslo_train.json.gzip'

    # Open the gzip file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        # Read and parse the JSON data
        data = json.load(f)


    print(len(data))
    print(data[0].keys())

    data_class = DataClass(data)


    # Path to save the pickled file
    pickle_file_path = './data/data_class.pkl'

    # Pickle the data
    with open(pickle_file_path, 'wb') as pf:
        pickle.dump(data_class, pf)

else:

    # Path to your pickle file
    pickle_file_path = './data/data_class.pkl'

    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as pf:
        data_class = pickle.load(pf)

print(data_class.URLs[0])

ixs = [0, 5, 7, 9, 11]

for curr_ix in ixs:

    curr_URL = data_class.URLs[curr_ix]
    split_curr_URL = curr_URL.split('/')
    print(split_curr_URL)
    useful_split_curr_URL = split_curr_URL[3:-2]
    print(useful_split_curr_URL)

    curr_authors = data_class.authors[curr_ix]
    print(curr_authors)
    useful_curr_authors = tuple(sorted(curr_authors))
    print(useful_curr_authors)

    curr_date = data_class.dates[curr_ix]
    print(curr_date)
    print(type(curr_date))
    year, month, day_hour = curr_date.split("-")
    day, hour = day_hour.split("T")
    hour = hour.split(":")[0]
    # conv to ints in one line
    year, month, day, hour = map(int, [year, month, day, hour])
    part_of_day = hour // 3
    print(year, month, day, hour, part_of_day)

    curr_num_of_figs = data_class.nums_of_figs[curr_ix]
    print(curr_num_of_figs)

    curr_lead = data_class.leads[curr_ix].split(" ")
    print(curr_lead)

    curr_topics = data_class.topics[curr_ix]
    print(curr_topics)

    curr_keywords = data_class.keywords[curr_ix]
    print(curr_keywords)

    curr_gpt_keywords = data_class.gpt_keywords[curr_ix]
    print(curr_gpt_keywords)

    curr_num_of_comments = data_class.num_of_comments[curr_ix]
    print(curr_num_of_comments)





more_than_one_topic = 0
less_than_one_topic = 0
for ix in range(data_class.length):
    curr_topics = data_class.topics[ix].split(" ")
    if len(curr_topics) > 1:
        more_than_one_topic += 1
    elif len(curr_topics) < 1:
        less_than_one_topic += 1
print("Num of articles with more than one topic:")
print(more_than_one_topic)
print("Num of articles with less than one topic:")
print(less_than_one_topic)





target_topics = ["gospodarstvo", "sport", "kultura", "zabava-in-slog"]
nums_by_topic = [0, 0, 0, 0, 0]
for ix in range(data_class.length):
    curr_topics = data_class.topics[ix]

    try:
        topic_ix = target_topics.index(curr_topics)
    except ValueError:
        topic_ix = 4
    
    nums_by_topic[topic_ix] += 1

print("Nums by topic:")
print(nums_by_topic)





target_topics = {}
for ix in range(data_class.length):
    curr_topics = data_class.topics[ix]
    
    if curr_topics not in target_topics:
        target_topics[curr_topics] = 0
    else:
        target_topics[curr_topics] += 1

print("Target topics:")
print(target_topics)

"""
{'sport': 4964, 'slovenija': 2775, 'kultura': 3490, 'zabava-in-slog': 3216,
 'kolumne': 62, 'svet': 3820, 'okolje': 722,
 'stevilke': 27, 'crna-kronika': 483, 'gospodarstvo': 1164, 'znanost-in-tehnologija': 239}
"""



import numpy as np
import matplotlib.pyplot as plt


target_topics = {}
for ix in range(data_class.length):
    curr_topics = data_class.topics[ix]
    curr_num_of_comments = data_class.num_of_comments[ix]
    
    if curr_topics not in target_topics:
        target_topics[curr_topics] = {}
    
    if curr_num_of_comments not in target_topics[curr_topics]:
        target_topics[curr_topics][curr_num_of_comments] = 1
    else:
        target_topics[curr_topics][curr_num_of_comments] += 1

prepared_graphs = {}
for topic, comment_dict in target_topics.items():
    max(comment_dict.values())
    graph = np.zeros((2, max(comment_dict.keys()) + 1))
    graph[0] = np.array(range(max(comment_dict.keys()) + 1))

    for num_comments, num_articles in comment_dict.items():
        graph[1][num_comments] = num_articles

    graph[1] = graph[1] / np.sum(graph[1])
    graph[1] = np.convolve(graph[1], np.ones(30)/30, 'same') / 10
    prepared_graphs[topic] = graph
    

plot_num = 0
for topic, graph in prepared_graphs.items():
    # plt.figure(plot_num)
    # plot_num += 1

    plt.plot(graph[0], graph[1], label=topic)
    # plt.title(f"Num of comments for topic: {topic}")
    # plt.show(block=False)
    
plt.legend(loc='upper center')
plt.show()
# input("Waiting...")
    


























class Article:

    def __init__(self, word_list):
        self.word_list = word_list

    def __eq__(self, other):
        return self.word_list == other.word_list
    


    def keep_only_acceptable_keywords(self, acceptable_keywords_list):
        new_word_list = []
        for keyword in self.word_list:
            if keyword in acceptable_keywords_list:
                new_word_list.append(keyword)
        self.word_list = new_word_list
    
    def keep_only_acceptable_keywords_set(self, acceptable_keywords_set):
        new_word_list = []
        for keyword in self.word_list:
            if keyword in acceptable_keywords_set:
                new_word_list.append(keyword)
        self.word_list = new_word_list
    

class Keywords:

    def __init__(self):
        self.keyword2article_count = dict()
    
    def __eq__(self, value: object) -> bool:
        return self.keyword2article_count == value.keyword2article_count
    
    def add_keyword(self, keyword):

        if keyword not in self.keyword2article_count:
            self.keyword2article_count[keyword] = 0

        self.keyword2article_count[keyword] += 1
    
    def add_article_keywords(self, article_list):

        for article in article_list:

            for keyword in article.word_list:
                self.add_keyword(keyword)
        
    
    def trim_keywords(self, threshold):

        keywords_to_del = []
        for keyword, count in self.keyword2article_count.items():
            if count < threshold:
                keywords_to_del.append(keyword)

        for keyword in keywords_to_del: 
            self.keyword2article_count.pop(keyword)
    
    # def get_keywords(self):
    #     return list(self.keyword2article_count.keys())




def tfidf(list_of_lists_of_words, word_repetition_cutoff=0):


    articles = []

    num_of_empty_articles = 0
    empty_articles_ixs = []

    for ix, word_list in enumerate(list_of_lists_of_words):

        word_list_to_keep = []

        keywords_unique = set()
        for keyword in word_list:
            # print(keyword)
            keyword_to_add = keyword.lower()
            if keyword_to_add not in keywords_unique:
                keywords_unique.add(keyword_to_add)
                word_list_to_keep.append(keyword_to_add)
        

        if len(word_list_to_keep) == 0:
            num_of_empty_articles += 1
            empty_articles_ixs.append(ix)

        articles.append(Article(word_list_to_keep))


    if True:
        print("num_of_empty_articles")
        print(num_of_empty_articles)

    keywords = Keywords()
    keywords.add_article_keywords(articles)

    acceptable_keyword2idf = dict()
    for keyword, article_count in keywords.keyword2article_count.items():
        if article_count >= word_repetition_cutoff:
            idf = np.log(len(articles) / article_count)
            acceptable_keyword2idf[keyword] = idf


    acceptable_keywords = list(acceptable_keyword2idf.keys())
    # OOOOOOOOOOOOOOOOOOOHHHHHHHHHHHHHHHHHHHHHHHHHH
    # ZDAAAAAAAAAAAAAAAAAAAAAAAJ RAZUMEM

    # S tem, ko sem odtranil vse besede, ki niso acceptable iz clankov,
    # sem spremenil stevilo besed v clankih, kar spremeni tf-idf.
    # V nalogi je pa napisano, da najprej naredimo tf-idf, potem pa odstranimo te besede, ki jih je manj kot 20.
    # Torej ce iz clankov ne odstranis teh besed, obdrzijo originalen length in tako je kot bi najprej naredil tf-idf.
    # In zato potem pride pravilno.

    # TO JE CULPRIT KI ME JE UBIJAL 2 DNI:
    # # ZAKAAAAAAJ!!?!?!?!?
    # for article in articles:
    #     article.keep_only_acceptable_keywords(acceptable_keywords)

    # TOLE NE POMAGA:
    # acceptable_keywords_set = set(acceptable_keywords)
    # for article in articles:
    #     article.keep_only_acceptable_keywords_set(acceptable_keywords_set)

    newly_empty_articles = 0
    empty_articles_after_cutoff_ixs = []

    article_keywords_tfs = np.zeros((len(articles), len(acceptable_keywords)))
    for ix, article in enumerate(articles):
        any_usable = False
        for keyword in article.word_list:
            if keyword in acceptable_keywords:
                article_keywords_tfs[ix, acceptable_keywords.index(keyword)] = 1/len(articles[ix].word_list)
                any_usable = True
            
        if not any_usable:
            newly_empty_articles += 1
            empty_articles_after_cutoff_ixs.append(ix)


    # article_keywords_tfs = np.delete(article_keywords_tfs, row_ixs_to_delete, axis=0)


    if True:
        print("newly_empty_articles")
        print(newly_empty_articles)

    articles_tfidf = article_keywords_tfs #.copy()
    for keyword in acceptable_keywords:
        articles_tfidf[:, acceptable_keywords.index(keyword)] *= acceptable_keyword2idf[keyword]
    
    return articles_tfidf, empty_articles_ixs, empty_articles_after_cutoff_ixs





