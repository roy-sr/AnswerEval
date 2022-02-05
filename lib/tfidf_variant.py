import nltk
import numpy as np
import re
import heapq
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer



class TFIDFVariant2(object):
    def __init__(self):
        super(TFIDFVariant2, self).__init__()
        self.vectorizer = TfidfVectorizer()
        self.source = []
    
    def name(self):
        return "TFIDFVariant2"

    def set_source(self, source):
        
        if type(source) is str:
            self.source = [source]
        elif type(source) is list:
            self.source = source
        else:
            raise NotImplementedError('Implementation error in tfidf input for input type ', type(source) )

    def get_tfidf_score(self, corpus ):
        
        for i in range(len(corpus )):
            corpus [i] = corpus [i].lower()
            corpus [i] = re.sub(r'\W',' ',corpus [i])
            corpus [i] = re.sub(r'\s+',' ',corpus [i])
            
        self.vectorizer.fit(corpus)
        
        return self.vectorizer.transform([' '.join(corpus)]).toarray().reshape(-1,1)
        
        
        
class TFIDFVariant(object):
    def __init__(self):
        super(TFIDFVariant, self).__init__()
        self.source = []
        self.article_text = []
    
    def name(self):
        return "TFIDFVariant"


    def set_source(self, source):
        
        if type(source) is str:
            self.source = [source]
        elif type(source) is list:
            self.source = source
        else:
            raise NotImplementedError('Implementation error in tfidf input for input type ', type(source) )

    def get_tfidf_score(self, corpus ):

        # article_text =  ' '.join(self.article_text)
        print('Starting tfidf')
        # corpus = nltk.sent_tokenize(article_text)

        corpus_word_tokenize = []
        for i in range(len(corpus )):
            corpus [i] = corpus [i].lower()
            corpus [i] = re.sub(r'\W',' ',corpus [i])
            corpus [i] = re.sub(r'\s+',' ',corpus [i])
            
            corpus_word_tokenize.append(nltk.word_tokenize(corpus [i]))

        df_corpus_word_tokenize = pd.DataFrame(corpus_word_tokenize,columns=['tokens'])
        df_corpus_word_tokenize['num'] = 1
        df_corpus_word_tokenize.set_index('Column_Index')
        df_corpus_word_tokenize = (df_corpus_word_tokenize.astype(str).explode('tokens') .groupby('tokens').size().reset_index(name='counts'))
        
        corpus_word_token = list(df_corpus_word_tokenize[['tokens','num']].groupby('tokens').groups.keys())
        corpus_word_count = df_corpus_word_tokenize[['tokens','num']].groupby('tokens').agg(['count'])['num'].values.tolist() 

        wordfreq = dict(zip(corpus_word_token, corpus_word_count))
        
        
        # wordfreq = {}  # keys are the unique words
        # for tokens in corpus_word_tokenize:
        #     for token in tokens:
        #         if token not in wordfreq.keys():
        #             wordfreq[token] = 1
        #         else:
        #             wordfreq[token] += 1


        word_idf_values = {}
        for token in wordfreq:
            doc_containing_word = 0
            for document_tokenized in corpus_word_tokenize:
                if token in document_tokenized:
                    doc_containing_word += 1
            word_idf_values[token] = np.log(len(corpus)/(1 + doc_containing_word))
            


        word_tf_values = {}
        article_text =  ' '.join(corpus)
        article_text_token = nltk.word_tokenize(article_text)

        df_article_text = pd.DataFrame(article_text_token, columns=['token'])
        df_article_text['num'] = 1


        article_text_token = list(df_article_text[['token','num']].groupby('token').groups.keys())
        article_text_freq = df_article_text[['token','num']].groupby('token').agg(['count'])['num'].values.tolist()

        word_tf_values = dict(zip(article_text_token,article_text_freq))


        tfidf_values_dict = {}
        for token in word_tf_values.keys():
            tfidf_sentences = []
            for tf_sentence in word_tf_values[token]:
                tf_idf_score = tf_sentence * word_idf_values[token]
                tfidf_sentences.append(tf_idf_score)
            tfidf_values_dict[token] = tfidf_sentences
            

        return tfidf_values_dict
    

    def get_tf_score(self, corpus ,freq= None):

        # article_text =  ' '.join(self.article_text)
        print('Starting tf')
        # corpus = nltk.sent_tokenize(article_text)

        corpus_word_tokenize = []
        for i in range(len(corpus )):
            corpus [i] = corpus [i].lower()
            corpus [i] = re.sub(r'\W',' ',corpus [i])
            corpus [i] = re.sub(r'\s+',' ',corpus [i])
            
            corpus_word_tokenize.append(nltk.word_tokenize(corpus [i]))


        wordfreq = {}  # keys are the unique words
        for tokens in corpus_word_tokenize:
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1


        if freq is not None:
            wordfreq = heapq.nlargest(200, wordfreq, key=wordfreq.get)
            
            word_tf_values = {}
            for token in wordfreq:
                sent_tf_vector = []
                for document_tokenized in corpus_word_tokenize:
                    doc_freq = 0
                    for word in document_tokenized:
                        if token == word:
                            doc_freq += 1
                    word_tf = doc_freq/len(document_tokenized)
                    sent_tf_vector.append(word_tf)
                word_tf_values[token] = sent_tf_vector
            

        else:

            article_text =  ' '.join(corpus)
            article_text_token = nltk.word_tokenize(article_text)

            df_article_text = pd.DataFrame(article_text_token, columns=['token'])
            df_article_text['num'] = 1


            article_text_token = list(df_article_text[['token','num']].groupby('token').groups.keys())
            article_text_freq = df_article_text[['token','num']].groupby('token').agg(['count'])['num'].values.tolist()
            
            max_article_text_freq = int(np.amax(article_text_freq))
            min_article_text_freq = int(np.amin(article_text_freq))

            word_tf_values = dict(zip(article_text_token,article_text_freq))


        return word_tf_values
    



     
if __name__ == "__main__":
    # tfidfVariant = TFIDFVariant()
    # tfidfVariant.set_source(['https://en.wikipedia.org/wiki/Natural_language_processing','https://en.wikipedia.org/wiki/Natural_language_processing'])
    # tfidfVariantOut = tfidfVariant.get_tfidf_score(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.", "In many offices, workers are given unrestricted access to the Web, allowing them to manage their own research.", "The use of search engines helps users to find information quickly.", "However, information published online may not always be reliable, due to the lack of authority-approval or a compulsory accuracy check before publication.", "Internet information lacks credibility as the Web's search engines do not have the abilities to filter and manage information and misinformation.", "This results in people having to cross-check what they read before using it for decision-making, which takes up more time.Viktor Mayer-Schönberger, author of Delete: The Virtue of Forgetting in the Digital Age, argues that everyone can be a \"participant\" on the Internet, where they are all senders and receivers of information. On the Internet, trails of information are left behind, allowing other Internet participants to share and exchange information. Information becomes difficult to control on the Internet."])
    # tfidfVariantOut = tfidfVariant.get_tf_score(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.", "In many offices, workers are given unrestricted access to the Web, allowing them to manage their own research.", "The use of search engines helps users to find information quickly.", "However, information published online may not always be reliable, due to the lack of authority-approval or a compulsory accuracy check before publication.", "Internet information lacks credibility as the Web's search engines do not have the abilities to filter and manage information and misinformation.", "This results in people having to cross-check what they read before using it for decision-making, which takes up more time.Viktor Mayer-Schönberger, author of Delete: The Virtue of Forgetting in the Digital Age, argues that everyone can be a \"participant\" on the Internet, where they are all senders and receivers of information. On the Internet, trails of information are left behind, allowing other Internet participants to share and exchange information. Information becomes difficult to control on the Internet."])
    # print(tfidfVariantOut)
    
    tfidfVariant2 = TFIDFVariant2()
    with open('interviewAssesment/refAnsFiles/ref_text_to_process/ref_text_to_process.json', 'r') as fp:
        list_in = json.load( fp)
        
    tfidfVariant2.vectorizer.fit(list_in)
    
    # tfidfVariant2.vectorizer.fit(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.", "In many offices, workers are given unrestricted access to the Web, allowing them to manage their own research.", "The use of search engines helps users to find information quickly.", "However, information published online may not always be reliable, due to the lack of authority-approval or a compulsory accuracy check before publication.", "Internet information lacks credibility as the Web's search engines do not have the abilities to filter and manage information and misinformation.", "This results in people having to cross-check what they read before using it for decision-making, which takes up more time.Viktor Mayer-Schönberger, author of Delete: The Virtue of Forgetting in the Digital Age, argues that everyone can be a \"participant\" on the Internet, where they are all senders and receivers of information. On the Internet, trails of information are left behind, allowing other Internet participants to share and exchange information. Information becomes difficult to control on the Internet."])
    # print(tfidfVariant2.vectorizer.get_feature_names())
    # print(tfidfVariant2.vectorizer.transform(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information."]).toarray().reshape(-1,1))
    # print('----')
    # print(tfidfVariant2.vectorizer.transform(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.","the World Wide Web has provided access to billions of pages of information."]).toarray().reshape(-1,1))
    # print('----')
    # print(tfidfVariant2.vectorizer.transform([' '.join(["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.", "In many offices, workers are given unrestricted access to the Web, allowing them to manage their own research.", "The use of search engines helps users to find information quickly.", "However, information published online may not always be reliable, due to the lack of authority-approval or a compulsory accuracy check before publication.", "Internet information lacks credibility as the Web's search engines do not have the abilities to filter and manage information and misinformation.", "This results in people having to cross-check what they read before using it for decision-making, which takes up more time.Viktor Mayer-Schönberger, author of Delete: The Virtue of Forgetting in the Digital Age, argues that everyone can be a \"participant\" on the Internet, where they are all senders and receivers of information. On the Internet, trails of information are left behind, allowing other Internet participants to share and exchange information. Information becomes difficult to control on the Internet."])]).toarray().reshape(-1,1))
    print('----')
    
    # list_in = ["In addition to e-mail, the World Wide Web has provided access to billions of pages of information.", "In many offices, workers are given unrestricted access to the Web, allowing them to manage their own research.", "The use of search engines helps users to find information quickly.", "However, information published online may not always be reliable, due to the lack of authority-approval or a compulsory accuracy check before publication.", "Internet information lacks credibility as the Web's search engines do not have the abilities to filter and manage information and misinformation.", "This results in people having to cross-check what they read before using it for decision-making, which takes up more time.Viktor Mayer-Schönberger, author of Delete: The Virtue of Forgetting in the Digital Age, argues that everyone can be a \"participant\" on the Internet, where they are all senders and receivers of information. On the Internet, trails of information are left behind, allowing other Internet participants to share and exchange information. Information becomes difficult to control on the Internet."]
    

    
    
    tfidf_score = tfidfVariant2.get_tfidf_score(list_in).squeeze(1)
    print(tfidf_score.shape)
    print(np.array(tfidfVariant2.vectorizer.get_feature_names()).shape)
    print(dict(zip(np.array(tfidfVariant2.vectorizer.get_feature_names()), tfidf_score)))