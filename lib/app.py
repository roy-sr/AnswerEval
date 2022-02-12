import os
import nltk
from nltk.corpus import wordnet
import string
import numpy as np
from sklearn.preprocessing import normalize


try:
    from lib.constants import *
    from lib.keywords_extraction import *
    from lib.utils import *
    from lib.use import *
    from lib.interface import *
    from lib.google_search import *
    from lib.wiki_data import *
    from lib.tfidf_variant import *
    from lib.haystack_app import *

except:
    from constants import *
    from keywords_extraction import *
    from utils import *
    from use import *
    from interface import *
    from google_search import *
    from wiki_data import *
    from tfidf_variant import *
    from haystack_app import *




class App(object):
    def __init__(self, USE = None):
        super(App, self).__init__()
        if USE is not None:
            self.use = USE
        self.interfaceObj = ApiInterface()
        self.querygoogle = QueryGoogle(tld='co.in',user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582') #tld='com', lang='en', tbs='0', safe='off', num=10, start=0, stop=10, pause=2, country='', extra_params=None, user_agent=None, verify_ssl=True
        self.wikidata = WikiData()
        self.tfidfVariant = TFIDFVariant2()
        
        # self.haystackapp = Haystack_app()
        # self.temp = []

    
    def name(self):
        return "App"

# synonym
    def get_synonym(self,text_to_process ):
        synonyms = []
        
        try:
            for syn in wordnet.synsets(text_to_process):
                for lm in syn.lemmas():
                        synonyms.append(lm.name())#adding into synonyms
            return list(set(synonyms))
        
        # text_to_process = travel
        # return ['trip', 'locomote', 'jaunt', 'change_of_location', 'go', 'traveling', 'travelling', 'locomotion', 'travel', 'move', 'journey', 'move_around']
        
        except Exception as e:
            print(e)
            return []
   
# Content words
    def get_content_words(self,text_to_process ):
        parseResult = self.interfaceObj.get_stanfordNLU_data(text_to_process,properties={"annotators": "tokenize,ssplit,pos,lemma,ner", "outputFormat": "json"})
        
        if len(parseResult['sentences']) < 1:
            return None
        
        content_index , content_originaltext, content_words, content_words_lemma, content_words_ner, content_words_pos = [], [], [], [], [], []
        all_words , all_words_lemma = [], []
        for sentence_dict in parseResult['sentences']:
            for sentence_dict_tokenDict in sentence_dict['tokens']:
                
                if sentence_dict_tokenDict['ner'] in ['PERSON', 'ORGANIZATION', 'LOCATION','MONEY', 'PERCENT', 'DATE', 'TIME', 'MISC']:
                    content_words_ner.append(sentence_dict_tokenDict['lemma'] +'###'+ sentence_dict_tokenDict['ner'])
                
                if sentence_dict_tokenDict['pos'][:1] in  ['N','V','R','J']: #Content words POS are Noun, Verb, Adjective and Adverb. See below for full list 
                    
                    content_index.append(sentence_dict_tokenDict['index'])
                    content_originaltext.append(sentence_dict_tokenDict['originalText'])
                    content_words.append(sentence_dict_tokenDict['word'])
                    content_words_lemma.append(sentence_dict_tokenDict['lemma'])
                    content_words_pos.append(sentence_dict_tokenDict['pos'])
                
                
                if sentence_dict_tokenDict['originalText'] not in list(string.punctuation):
                    all_words.append(sentence_dict_tokenDict['originalText'] )
                    all_words_lemma.append(sentence_dict_tokenDict['lemma'] )
    
        return content_index, content_originaltext, content_words, content_words_lemma, content_words_pos, content_words_ner, all_words , all_words_lemma              
                
# demoting words
    def demote_words(self, textToDemote_list, usingtext_list  ):
        
        # if type(textToDemote) is str:
        #     _, textToDemote, _, _ = self.get_content_words(textToDemote)
        # if type(usingtext) is str:
        #     _, usingtext, _, _ = self.get_content_words(usingtext)
                
        return list(set(textToDemote_list) - set(usingtext_list))
    
# keywords
        
    def get_content_keywords(self,text_to_process ):
        keywords_extraction = KeywordsExtraction()
        # keywords = list(set(keywords_extraction.get_keyword(text_to_process) + keywords_extraction.get_aws_keyword(text_to_process)))
        keywords = keywords_extraction.get_keyword(text_to_process)

        keywords_vectors = self.use.get_vectors(keywords)
        text_to_process_vectors = self.use.get_vectors([text_to_process]) 
        
        keywords_cosine_sim = []
        for keywords_vectors_item in keywords_vectors:
            keywords_cosine_sim.append(cosine_similarity(keywords_vectors_item, text_to_process_vectors))


        keywords_cosine_sim_std = np.std(keywords_cosine_sim)
    
        largest_indices = np.argsort(-1*np.array(keywords_cosine_sim))
        
        keywords = list(map(keywords.__getitem__, largest_indices))
        keywords_cosine_sim = list(map(keywords_cosine_sim.__getitem__, largest_indices))
        keywords_cosine_sim = np.array(keywords_cosine_sim)[np.array(keywords_cosine_sim) > (keywords_cosine_sim[0] - (1*keywords_cosine_sim_std))]

        return keywords[:len(keywords_cosine_sim)+1]
        
    def get_keyword_sim_index(self,ref_text_to_process, eval_text_to_process ):
        
        ref_text_to_process_keywords = self.get_content_keywords(ref_text_to_process ) #; print(ref_text_to_process_keywords)
        ref_text_to_process_keywords_vectors = self.use.get_vectors(ref_text_to_process_keywords)
        
        
        eval_text_to_process_keywords = self.get_content_keywords(eval_text_to_process ) 
        eval_text_to_process_keywords_vectors = self.use.get_vectors(eval_text_to_process_keywords)
        
        # ref_eval_vector_combinations = itertools.product(ref_text_to_process_keywords_vectors,eval_text_to_process_keywords_vectors)
        
        # ref_eval_cosine_similarity = []
        # for ref_eval_vector_combinations_item in ref_eval_vector_combinations:
        #     ref_eval_cosine_similarity.append(cosine_similarity(ref_eval_vector_combinations_item[0], ref_eval_vector_combinations_item[1]))
        
        # ref_eval_cosine_similarity_std = np.std(ref_eval_cosine_similarity)

        # return np.array(ref_eval_cosine_similarity)[np.array(ref_eval_cosine_similarity) > (3*ref_eval_cosine_similarity_std) ].mean()
        
        ref_eval_cosine_similarity = []
        for ref_text_to_process_keywords_vectors_item in ref_text_to_process_keywords_vectors:
            for eval_text_to_process_keywords_vectors_item in eval_text_to_process_keywords_vectors:
                ref_eval_cosine_similarity.append(cosine_similarity(ref_text_to_process_keywords_vectors_item, eval_text_to_process_keywords_vectors_item))
        
        ref_eval_cosine_similarity_std = np.std(ref_eval_cosine_similarity) 
        
        try:
            sim_score = np.array(ref_eval_cosine_similarity)[np.array(ref_eval_cosine_similarity) > (3*ref_eval_cosine_similarity_std) ]
            
            if len(sim_score) > 0:
                sim_score = sim_score.mean()
            else:
                sim_score = 0
        except:
            sim_score = 0
            
        ref_eval_cosine_similarity_array = np.array(ref_eval_cosine_similarity).reshape(len(ref_text_to_process_keywords),len(eval_text_to_process_keywords))

        ref_eval_recall  = np.amax(ref_eval_cosine_similarity_array, axis=1) / np.sum(ref_eval_cosine_similarity_array, axis=1)
        
        ref_eval_precission = np.amax(ref_eval_cosine_similarity_array, axis=0) / np.sum(ref_eval_cosine_similarity_array, axis=0)

        ref_eval_F1 = 2 * ((np.array(ref_eval_recall).sum() * np.array(ref_eval_precission).sum() )/(np.array(ref_eval_recall).sum() + np.array(ref_eval_precission).sum()))
        
        return sim_score, ref_eval_F1

    # Google Search
    def search_google(self, text_to_search, search_site = None):
        
        if search_site is not None:
            self.querygoogle.set_source(search_site)
            
        return self.querygoogle.search_google_unique_URL(text_to_search)

    # Get wiki corpus
    def generate_wiki_corpora(self, keyWord_list):
        
        if type(keyWord_list) is not list:
            raise NotImplementedError('Implementation error in generate_wiki_corpora input for input type ', type(keyWord_list) )
        
        wiki_corpus = []
        wiki_url_repo = []
        for keyword in keyWord_list:
            wiki_url = self.search_google(keyword, 'wikipedia.org')
            
            for wiki_url_item in wiki_url:
                
                if wiki_url_item not in wiki_url_repo:
                    
                    if 'Category:' not in wiki_url_item:
                        wiki_url_repo.append(wiki_url_item)
                        page_py = self.wikidata.get_wikidata(wiki_url_item)
                        wiki_corpus.append(page_py.title + page_py.summary + page_py.text)
                        # print(wiki_url_item)
                        
                        try:
                            for title in sorted(page_py.links.keys()):
                                page_py_link = self.wikidata.get_wikidata(title)
                                
                                if (page_py_link.fullurl not in wiki_url_repo) \
                                        and 'Category:' not in page_py_link.fullurl:
                                            
                                    wiki_url_repo.append(page_py_link.fullurl)
                                    # print(page_py_link.fullurl)
                                    wiki_corpus.append(page_py_link.title + page_py_link.summary + page_py_link.text)
                        except:
                            pass
                    
            wiki_url_repo = list(set(wiki_url_repo))
            # print(len(wiki_url_repo))
        return wiki_corpus


    def save_ans_data(self, ansID ,ansType, ansObj):
        
        if ansType == 'ref':
            with open('refAnsFiles/' + ansID + '.json', 'w') as fp:
                json.dump(ansObj,fp)

    def read_ans_data(self, ansID ,ansType):
        
        if ansType == 'ref':
            try:
                with open('refAnsFiles/' + ansID + '.json', 'r') as fp:
                    return json.load(fp)
            except Exception as e:
                print('read_ans_data exception ', e)
                return {}

    def get_tfidf(self,refAnstext_keywords):
        
        
        wikiCorpora = self.generate_wiki_corpora(refAnstext_keywords)
        
        
        for wikiCorpora_item in wikiCorpora:
            
            
            wikiCorpora_content_words_lemma = []
            wikiCorpora_item_sent_list = nltk.sent_tokenize(wikiCorpora_item)
            
            for wikiCorpora_item_sent_list_item in wikiCorpora_item_sent_list:
                try:
                    wikiCorpora_item_sent_list_item = json.dumps(str(wikiCorpora_item_sent_list_item.encode("ascii", "ignore").decode()).strip().replace('%',''))
                    wikiCorpora_parseResult = self.interfaceObj.get_stanfordNLU_data(wikiCorpora_item_sent_list_item,properties={"annotators": "lemma", "outputFormat": "json"})
                
                    if len(wikiCorpora_parseResult['sentences']) < 1:
                        self.tfidfVariantOut.append([])
                    
                    
                    for wikiCorpora_sentence_dict in wikiCorpora_parseResult['sentences']:
                        for sentence_dict_tokenDict in wikiCorpora_sentence_dict['tokens']:
                            
                            wikiCorpora_content_words_lemma.append(sentence_dict_tokenDict['lemma'])
                            
                            # if sentence_dict_tokenDict['pos'][:1] in  ['N','V','R','J']: #Content words POS are Noun, Verb, Adjective and Adverb. See below for full list 
                            #     wikiCorpora_content_words_lemma.append(sentence_dict_tokenDict['lemma'])

                except Exception as e:
                    print(e, wikiCorpora_item_sent_list_item)
                    # pass
            
            self.tfidfVariantOut.append(' '.join(wikiCorpora_content_words_lemma))
               
        tfidfVariantOut = self.tfidfVariant.get_tfidf_score(self.tfidfVariantOut)
        # tfidfVariantOut = self.tfidfVariant.get_tf_score(self.tfidfVariantOut)

        return tfidfVariantOut
    
    def get_content_words_context_score(self, refAnstext, content_index, content_words_lemma , \
                                        tfidfVariantOut, synonyms_dict=None, max_article_text_freq= None, min_article_text_freq = None):
        
        """
            This function calculates the context score of the content word w.r.t the sentence using cosine similarity
        """
        sent_tokenize = nltk.sent_tokenize(refAnstext)
        
        # Get synonyms of ref content words
        # Get context similarity of ref content words in the sentence
        
        # synonyms_dict = {}
        content_word_context_score = []
        old_content_index_item = 1
        sent_tokenize_idx = 0
        sent_tokenize_item_vector = 0
        
        for content_index_item, content_words_lemma_item in zip(content_index, content_words_lemma):
            
            if synonyms_dict is not None:
                synonyms_dict[content_words_lemma_item] = self.get_synonym(content_words_lemma_item)
            
            content_words_lemma_vectors = self.use.get_vectors([content_words_lemma_item])

            # if content_words_lemma_item in tfidfVariantOut:
            #     content_words_lemma_vectors = content_words_lemma_vectors *tfidfVariantOut[content_words_lemma_item][0]
                
            if content_index_item <= old_content_index_item: # next sentence
 
                sent_tokenize_item_vector = self.use.get_vectors([sent_tokenize[sent_tokenize_idx]]) 
                sent_tokenize_idx +=1
                
            old_content_index_item = content_index_item
            
            cosine_sim = cosine_similarity(content_words_lemma_vectors, sent_tokenize_item_vector)
            if content_words_lemma_item in tfidfVariantOut:
                if max_article_text_freq is None or min_article_text_freq is None:
                    cosine_sim = cosine_sim *tfidfVariantOut[content_words_lemma_item] 
                else:
                    cosine_sim = cosine_sim *(tfidfVariantOut[content_words_lemma_item] - min_article_text_freq) / max_article_text_freq
                    
                
            content_word_context_score.append(cosine_sim)

        if synonyms_dict is not None:
            return synonyms_dict, content_word_context_score
        else:
            return content_word_context_score
    """
        Function to process intermediate data for reference answer. It should be batch process.
    """
    def prepare_reference_ans(self,refQAID,  refAnstext, ref_qtext_to_process = None):
        
        # Get content words
        content_index, content_originaltext, content_words, content_words_lemma,  \
                                                content_words_pos, content_words_ner, all_words, all_words_lemma   = self.get_content_words(refAnstext)


        
        if ref_qtext_to_process:
            qtext_content_index, qtext_content_originaltext, qtext_content_words, qtext_content_words_lemma,  \
                                        qtext_content_words_pos, qtext_content_words_ner, qtext_all_words, qall_words_lemma   = self.get_content_words(ref_qtext_to_process)
                     
        # Get tfIdf matrix                                                                    
        refAnstext_keywords = self.get_content_keywords(refAnstext)
        self.tfidfVariantOut = []
        
        if not os.path.exists('refAnsFiles/' + refQAID + '/' + refQAID + '.json'):
        
            wikiCorpora = self.generate_wiki_corpora(refAnstext_keywords)
            
            
            for wikiCorpora_item in wikiCorpora:
                
                
                wikiCorpora_content_words_lemma = []
                wikiCorpora_item_sent_list = nltk.sent_tokenize(wikiCorpora_item)
                
                for wikiCorpora_item_sent_list_item in wikiCorpora_item_sent_list:
                    try:
                        wikiCorpora_item_sent_list_item = json.dumps(str(wikiCorpora_item_sent_list_item.encode("ascii", "ignore").decode()).strip().replace('%',''))
                        wikiCorpora_parseResult = self.interfaceObj.get_stanfordNLU_data(wikiCorpora_item_sent_list_item,properties={"annotators": "lemma", "outputFormat": "json"})
                    
                        if len(wikiCorpora_parseResult['sentences']) < 1:
                            self.tfidfVariantOut.append([])
                        
                        
                        for wikiCorpora_sentence_dict in wikiCorpora_parseResult['sentences']:
                            for sentence_dict_tokenDict in wikiCorpora_sentence_dict['tokens']:
                                
                                wikiCorpora_content_words_lemma.append(sentence_dict_tokenDict['lemma'])
                                
                                # if sentence_dict_tokenDict['pos'][:1] in  ['N','V','R','J']: #Content words POS are Noun, Verb, Adjective and Adverb. See below for full list 
                                #     wikiCorpora_content_words_lemma.append(sentence_dict_tokenDict['lemma'])

                    except Exception as e:
                        print(e, wikiCorpora_item_sent_list_item)
                        # pass
                
                self.tfidfVariantOut.append(' '.join(wikiCorpora_content_words_lemma))
            
            if not os.path.exists('refAnsFiles/' + refQAID + '/'):
                os.mkdir('refAnsFiles/' + refQAID + '/')
            
            with open('refAnsFiles/' + refQAID + '/' + refQAID + '.json', 'w') as fp:
                json.dump( self.tfidfVariantOut,fp)

        else:
            with open('refAnsFiles/' + refQAID + '/' + refQAID + '.json', 'r') as fp:
                 self.tfidfVariantOut = json.load( fp)
                      
        # tfidfVariantOut = self.tfidfVariant.get_tfidf_score(self.tfidfVariantOut)
        # tfidfVariantOut = self.tfidfVariant.get_tf_score(self.tfidfVariantOut)
        self.tfidfVariant.vectorizer.fit(self.tfidfVariantOut)
        self.tfidfVariantOut = self.tfidfVariant.get_tfidf_score(self.tfidfVariantOut).squeeze(1)
        
        tfidfVariantOut_min = np.array(self.tfidfVariantOut).min()        
        self.tfidfVariantOut = (self.tfidfVariantOut - tfidfVariantOut_min) / (np.array(self.tfidfVariantOut).max() - tfidfVariantOut_min)

        self.tfidfVariantOut = dict(zip(np.array(self.tfidfVariant.vectorizer.get_feature_names()) , self.tfidfVariantOut))
        
        synonyms_dict, content_word_context_score = self.get_content_words_context_score(refAnstext, content_index, content_words_lemma , self.tfidfVariantOut, synonyms_dict = {})

        refAns_dataDict = {'refAns_content_originaltext':content_originaltext,'refAns_content_words':content_words, 'refAns_content_words_lemma':content_words_lemma,\
                                'refAns_content_words_pos':content_words_pos, 'refAns_content_words_ner':content_words_ner ,'refAns_all_words':all_words, \
                                    'refAns_synonyms_dict':synonyms_dict ,'refAns_tfidfVariantOut':self.tfidfVariantOut, 'all_words_lemma':all_words_lemma, \
                                    'refAns_content_word_context_score':content_word_context_score, 'refAns_keywords':refAnstext_keywords, \
                                'qtext_content_index':qtext_content_index, 'qtext_content_originaltext':qtext_content_originaltext, \
                                'qtext_content_words':qtext_content_words, 'qtext_content_words_lemma':qtext_content_words_lemma, 'qall_words_lemma':qall_words_lemma, \
                                    'qtext_content_words_pos':qtext_content_words_pos, 'qtext_content_words_ner':qtext_content_words_ner, \
                                        'qtext_all_words':qtext_all_words}
                                    
                            
        
        self.save_ans_data(ansID = refQAID ,ansType = 'ref', ansObj = refAns_dataDict)

    """
        Function to process intermediate data for reference answer. It should be batch process.
    """
    def evaluate_candidate_ans(self,refAnsID, candAnsID, candAnstext, haystack_coeff = 1.75):  
        
        # self.temp.append('================================')
        # self.temp.append(candAnsID)
        # self.temp.append('-------------------------------')
        
        refAns_dataDict = self.read_ans_data(refAnsID, ansType='ref')
        """
        {'refAns_content_originaltext':content_originaltext,'refAns_content_words':content_words, 'refAns_content_words_lemma':content_words_lemma,\
                                'refAns_content_words_pos':content_words_pos, 'refAns_content_words_ner':content_words_ner ,'refAns_all_words':all_words, \
                                    'refAns_synonyms_dict':synonyms_dict ,'refAns_tfidfVariantOut':tfidfVariantOut, 'all_words_lemma':all_words_lemma, \
                                        'refAns_content_word_context_score':content_word_context_score, 'refAns_keywords':refAnstext_keywords, \
                                'qtext_content_index':qtext_content_index, 'qtext_content_originaltext':qtext_content_originaltext, \
                                'qtext_content_words':qtext_content_words, 'qtext_content_words_lemma':qtext_content_words_lemma, 'qall_words_lemma':qall_words_lemma, \
                                    'qtext_content_words_pos':qtext_content_words_pos, 'qtext_content_words_ner':qtext_content_words_ner, \
                                        'qtext_all_words':qtext_all_words}
        """
        refAns_keywords = refAns_dataDict['refAns_keywords']
        tfidfVariantOut = refAns_dataDict['refAns_tfidfVariantOut']
        
        # max_article_text_freq = np.array(list(tfidfVariantOut.values())).max()
        # min_article_text_freq = np.array(list(tfidfVariantOut.values())).min()
        
        # candAns_content_index, candAns_content_originaltext, candAns_content_words, candAns_content_words_lemma,  \
        #                                 candAns_content_words_pos, candAns_content_words_ner, candAns_all_words, \
        #                                     candAns_all_words_lemma   = self.get_content_words(candAnstext)
        
        candAns_content_index, _, _, candAns_content_words_lemma, _, _, candAns_all_words, _   = self.get_content_words(candAnstext)
        
        
        # out                                
        # sim_keyword_score , sim_keyword_F1 =  self.get_keyword_sim_index(' '.join(candAns_all_words), candAnstext )
        # sim_keyword_score = round(sim_keyword_score, 2)
        # print('sim_keyword_score : ', sim_keyword_score)
        # self.temp.append('sim_keyword_score')
        # self.temp.append(sim_keyword_score)
        # print('sim_keyword_F1 : ', sim_keyword_F1)
        #----------------------------------------------------------------------------------------------#
        
        
        ### Weighted Recall
        
        ref_word_coverage = 0
        ref_word_coverage_list = [0]
        cand_word_coverage = 0
        cand_word_coverage_list = [0]
        for refAns_synonyms_dict_key, refAns_synonyms_dict_val in refAns_dataDict['refAns_synonyms_dict'].items(): # key is lemma of the word, so the tfidf words are
            
            if (refAns_synonyms_dict_key in tfidfVariantOut) and (refAns_synonyms_dict_key not in refAns_dataDict['qall_words_lemma']):
                ref_word_coverage_list.append(tfidfVariantOut[refAns_synonyms_dict_key] )
                # ref_word_coverage_list.append((tfidfVariantOut[refAns_synonyms_dict_key]- min_article_text_freq )/ max_article_text_freq )

            
            ref_word_coverage = ref_word_coverage + 1
            
            common_lemma_list = list(set(refAns_synonyms_dict_val).intersection(candAns_content_words_lemma))
            common_lemma_list = self.demote_words(common_lemma_list, refAns_dataDict['qall_words_lemma']  )
            
            if len(common_lemma_list) > 0:
                
                if refAns_synonyms_dict_key in tfidfVariantOut:
                    
                    cand_word_coverage_list.append(tfidfVariantOut[refAns_synonyms_dict_key] )
                    # cand_word_coverage_list.append((tfidfVariantOut[refAns_synonyms_dict_key] - min_article_text_freq )/ max_article_text_freq)

                
                cand_word_coverage = cand_word_coverage + 1

                
        # Out 
        word_coverage =  cand_word_coverage / ref_word_coverage
        word_coverage = round(word_coverage, 2)
        # print('word_coverage : ', word_coverage)
        # self.temp.append('word_coverage')
        # self.temp.append(word_coverage)
        #----------------------------------------------------------------------------------------------#
        
        
        if len(ref_word_coverage_list) > 1:
            del ref_word_coverage_list[0]
        if len(cand_word_coverage_list) > 1:
            del cand_word_coverage_list[0]
        # Out     
        # word_coverage_context =  word_coverage * np.array(cand_word_coverage_list).mean() / np.array(ref_word_coverage_list).mean()
        word_coverage_context =  word_coverage * np.array(cand_word_coverage_list).sum() / np.array(ref_word_coverage_list).sum()
        word_coverage_context = round(word_coverage_context, 2)
        # print('word_coverage_context : ', word_coverage_context)   
        # self.temp.append('word_coverage_context')
        # self.temp.append(word_coverage_context)         
        #----------------------------------------------------------------------------------------------#

        candAns_content_word_context_score = self.get_content_words_context_score(candAnstext, candAns_content_index, candAns_content_words_lemma , \
                                                    tfidfVariantOut)
        
        # # Out 
        sim_context_score = np.array(candAns_content_word_context_score).mean()/ np.array(refAns_dataDict['refAns_content_word_context_score']).mean()
        # sim_context_score = np.array(candAns_content_word_context_score).sum()/ np.array(refAns_dataDict['refAns_content_word_context_score']).sum()
        sim_context_score = round(sim_context_score, 2)
        # print('sim_context_score : ', sim_context_score)
        # self.temp.append('sim_context_score')
        # self.temp.append(sim_context_score)
        # print(' Final Score : ', ((sim_keyword_score * 25) + (word_coverage * 25) +  (word_coverage_context * 25) +(sim_context_score * 25 ) ))
        
        #----------------------------------------------------------------------------------------------#
        
        queries = 'what is ' + ' and '.join(refAns_keywords)
        data_dict = {'Question':candAnstext}
        
        haystackapp = Haystack_app()
        haystack_score = haystackapp.evaluator(data_dict, queries)['Question']
        # print('haystack_score : ', haystack_score)
        # self.temp.append('haystack_score')
        # self.temp.append(haystack_score)
        
        if haystack_coeff is None: # calculate the haystack_coeff
            z = (100 - (word_coverage_context * 50))
            # haystack_coeff = (1/((1/((((z/50)/2))/ (haystack_score * sim_context_score)))/haystack_score))* sim_context_score
            haystack_coeff = -(z * sim_context_score) / ((2* haystack_score * sim_context_score) - (haystack_score * z) )
            
 
        haystack_score *= haystack_coeff
        context_score = 2* (haystack_score * sim_context_score)/ (haystack_score + sim_context_score)
        
        # finale = ((sim_keyword_score * 25) + (word_coverage * 25) +  (word_coverage_context * 25) +(context_score * 25 ) )
        finale = ((word_coverage_context * 50) +(context_score * 50 ) )
        print(' Final Score : ', finale)

        # self.temp.append('finale')
        # self.temp.append(str(candAnsID) + ' score: ' + str(finale))
        
        return haystack_coeff , finale

        
        
        
if __name__ == "__main__":
    
    finale_score_list = []
    use = USE()
    app = App(use)
    
    # print(app.get_synonym('Natural'))
    ref_qtext_to_process = 'What is natural language procesing ?'
    ref_text_to_process  = "Natural language processing (NLP) is a subfield of Artificial Intelligence (AI). This is a widely used technology for personal assistants that are used in various business fields/areas. This technology works on the speech provided by the user, breaks it down for proper understanding and processes accordingly. This is a very recent and effective approach due to which it has a really high demand in today’s market. Natural Language Processing is an upcoming field where already many transitions such as compatibility with smart devices, interactive talks with a human have been made possible. Knowledge representation, logical reasoning, and constraint satisfaction were the emphasis of AI applications in NLP. Here first it was applied to semantics and later to the grammar. In the last decade, a significant change in NLP research has resulted in the widespread use of statistical approaches such as machine learning and data mining on a massive scale. The need for automation is never ending courtesy of the amount of work required to be done these days. NLP is a very favourable, but aspect when it comes to automated applications. The applications of NLP have led it to be one of the most sought-after methods of implementing machine learning. Natural Language Processing (NLP) is a field that combines computer science, linguistics, and machine learning to study how computers and humans communicate in natural language. The goal of NLP is for computers to be able to interpret and generate human language. This not only improves the efficiency of work done by humans but also helps in interacting with the machine. NLP bridges the gap of interaction between humans and electronic devices."
    # content_keywords = app.get_content_keywords(ref_text_to_process)
    # print(content_keywords)
    
    # eval_text_to_process0 = "Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can."
    # eval_text_to_process1 = "Neuro-linguistic programming is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s."
    # eval_text_to_process2 = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The goal is a computer capable of \"understanding\" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence. "
    # eval_text_to_process3 = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with computer-human language interactions, specifically how to program computers to process and analyze large amounts of natural language data. The goal is to create a computer that can \"understand\" the contents of documents, including the contextual nuances of the language used in them.The technology can then extract information and insights from the documents, as well as categorize and organize the documents themselves. Natural language processing problems frequently involve speech recognition, natural language comprehension, and natural language generation. Natural language processing dates back to the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" in which he proposed what is now known as the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time was not considered a criterion of intelligence. "
    # eval_text_to_process4 = "Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that makes human language intelligible to machines. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech."
    # eval_text_to_process5 = "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."
    # eval_text_to_process55 = "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."
    
    data_dict = {

            "eval_text_to_process0.txt" : "Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.",
            "eval_text_to_process1.txt" : "Neuro-linguistic programming is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s.",
            "eval_text_to_process2.txt" : "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The goal is a computer capable of \"understanding\" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence. ",
            "eval_text_to_process3.txt" : "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with computer-human language interactions, specifically how to program computers to process and analyze large amounts of natural language data. The goal is to create a computer that can \"understand\" the contents of documents, including the contextual nuances of the language used in them.The technology can then extract information and insights from the documents, as well as categorize and organize the documents themselves. Natural language processing problems frequently involve speech recognition, natural language comprehension, and natural language generation. Natural language processing dates back to the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" in which he proposed what is now known as the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time was not considered a criterion of intelligence. ",
            "eval_text_to_process4.txt" : "Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that makes human language intelligible to machines. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech.",
            "eval_text_to_process5.txt" : "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.",
            "eval_text_to_process55.txt" : "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."
        
    }
    
    
    # sim_idx, sim_F1 =  app.get_keyword_sim_index(ref_text_to_process, eval_text_to_process2 )
    # print(sim_idx)
    # print(sim_F1)
    # import json
    # print(json.dumps(ref_text_to_process))
    
    
    # content_words1, content_words_lemma1, content_words_pos1, all_words1 = app.get_content_words(ref_text_to_process)
    # content_words2, content_words_lemma2, content_words_pos2, all_words2 = app.get_content_words(eval_text_to_process)
    
    # print(len(content_words1), len(content_words_lemma1), len(content_words_pos1), len(all_words1))
    # print(len(content_words2), len(content_words_lemma2), len(content_words_pos2), len(all_words2))
    
    # print(len(app.demote_words(textToDemote = ' jumped, trumph',usingtext = ' jump')))
    
    # print(len(app.demote_words(textToDemote = content_words_lemma2,usingtext = content_words_lemma1)))
    # print(len(app.demote_words(textToDemote = content_words_lemma1,usingtext = content_words_lemma2)))
    
    
    # for j in app.search_google('Natural language processing', 'wikipedia.org'):
    #     print(j)
    
    # page_py = app.wikidata.get_wikidata('https://en.wikipedia.org/wiki/Neuro-linguistic_programming')
    # print(page_py.title + page_py.summary + page_py.text)
    
    # wikiCorpora = app.generate_wiki_corpora(content_keywords)
    # tfidfVariantOut = app.tfidfVariant.get_tfidf_score(wikiCorpora)
    
    # app.prepare_reference_ans('refQ1', ref_text_to_process, ref_qtext_to_process)
    # print('ref ans processing complete')
    
    print(' ===================  ref_text_to_process ans processing start =================== ')    
    haystack_coeff, score = app.evaluate_candidate_ans(refAnsID='refQ1', candAnsID='ref_text_to_process', candAnstext=ref_text_to_process, haystack_coeff = None)
    finale_score_list.append('refQ1 ' + str(score))
    print(' ===================  ref_text_to_process ans processing complete =================== ') 
    print(haystack_coeff)
    print(' ===================  eval_text_to_process0 ans processing start =================== ')    
    _ , score= app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process0', candAnstext=data_dict["eval_text_to_process0.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process0 ' + str(score))
    print(' ===================  eval_text_to_process0 ans processing complete =================== ')     
    print(' ===================  eval_text_to_process1 ans processing start =================== ')    
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process1', candAnstext=data_dict["eval_text_to_process1.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process1 ' + str(score))
    print(' ===================  eval_text_to_process1 ans processing complete =================== ') 
    print(' ===================  eval_text_to_process2 ans processing start  =================== ')     
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process2', candAnstext=data_dict["eval_text_to_process2.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process2 ' + str(score))
    print(' ===================  eval_text_to_process2 ans processing complete =================== ')  
    print(' ===================  eval_text_to_process3 ans processing start  =================== ')  
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process3', candAnstext=data_dict["eval_text_to_process3.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process3 ' + str(score))
    print(' ===================  eval_text_to_process3 ans processing complete =================== ')  
    print(' ===================  eval_text_to_process4 ans processing start  =================== ')  
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process4', candAnstext=data_dict["eval_text_to_process4.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process4 ' + str(score))
    print(' ===================  eval_text_to_process4 ans processing complete =================== ')  
    print(' ===================  eval_text_to_process44 ans processing start  =================== ')  
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process44', candAnstext=data_dict["eval_text_to_process4.txt"] + data_dict["eval_text_to_process4.txt"], \
                                            haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process44 ' + str(score))
    print(' ===================  eval_text_to_process4 ans processing complete =================== ')  
    print(' ===================  eval_text_to_process5 ans processing start  =================== ')  
    _, score = app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process5', candAnstext=data_dict["eval_text_to_process5.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process5 ' + str(score))
    print(' ===================  eval_text_to_process5 ans processing complete =================== ')  
    print(' ===================  eval_text_to_process55 ans processing start  =================== ')  
    _, score= app.evaluate_candidate_ans(refAnsID='ref_text_to_process', candAnsID='eval_text_to_process55', candAnstext=data_dict["eval_text_to_process55.txt"], haystack_coeff = haystack_coeff)
    finale_score_list.append('eval_text_to_process55 ' + str(score))
    print(' ===================  eval_text_to_process55 ans processing complete =================== ')  
    
    for temp_item in finale_score_list:
        print(temp_item)
    
    # only tf
    
