
import numpy as np

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
    


    empty_articles_ixs.extend(empty_articles_after_cutoff_ixs)
    final_empty_articles_ixs = set(empty_articles_ixs)

    return articles_tfidf, final_empty_articles_ixs

