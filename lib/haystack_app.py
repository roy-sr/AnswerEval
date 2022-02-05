from haystack.nodes import FARMReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import ExtractiveQAPipeline

try:
    from lib.constants import *
except:
    from constants import *
    
class Haystack_app(object):
    def __init__(self):
        
        self.document_store = InMemoryDocumentStore(embedding_dim=128)
        self.reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
        

        
    def name(self):
        return "Haystack_app"

    def format_input(self, data_dict):

        data_out = [] # [{"meta": {"name": "eval_text_to_process0.txt" }, 'content': "Natural language processing (NLP) ... .."}]
        
        for data_dict_key, data_dict_value in data_dict.items():

            data_out_dict = {}
            
            data_out_name_dict = {}
            data_out_name_dict["name"] = data_dict_key

            data_out_dict["meta"] = data_out_name_dict
            data_out_dict["content"] = data_dict_value
            
            data_out.append(data_out_dict)
            
        return data_out

    def evaluator(self, data_dict, query):
        
        data_dict_len = len(data_dict) 
        # self.dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

        data_dict = self.format_input(data_dict)
        self.document_store.write_documents(data_dict)
        
        retriever = EmbeddingRetriever(document_store=self.document_store,
                                        embedding_model="yjernite/retribert-base-uncased",
                                        model_format="retribert")
        
        self.document_store.update_embeddings(retriever)
        
        self.pipe = ExtractiveQAPipeline(self.reader, retriever)
        
        prediction = self.pipe.run(query=query, params={"Retriever": {"top_k": data_dict_len }, "Reader": {"top_k": data_dict_len}})
        
        score_dict = {}
        for prediction_item in prediction['answers']:
            
            if prediction_item.meta['name'] not in score_dict:
                score_dict[prediction_item.meta['name']] = prediction_item.score
                
        return score_dict                         


if __name__ == "__main__":
    
    # data_dict = [
    #         {"meta": {"name": "eval_text_to_process0.txt" }, 'content': "Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can."},
    #         {"meta": {"name": "eval_text_to_process1.txt" }, 'content': "Neuro-linguistic programming is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s."},
    #         {"meta": {"name": "eval_text_to_process2.txt" }, 'content': "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The goal is a computer capable of \"understanding\" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence. "},
    #         {"meta": {"name": "eval_text_to_process3.txt" }, 'content': "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with computer-human language interactions, specifically how to program computers to process and analyze large amounts of natural language data. The goal is to create a computer that can \"understand\" the contents of documents, including the contextual nuances of the language used in them.The technology can then extract information and insights from the documents, as well as categorize and organize the documents themselves. Natural language processing problems frequently involve speech recognition, natural language comprehension, and natural language generation. Natural language processing dates back to the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" in which he proposed what is now known as the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time was not considered a criterion of intelligence. "},
    #         {"meta": {"name": "eval_text_to_process4.txt" }, 'content': "Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that makes human language intelligible to machines. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech."},
    #         {"meta": {"name": "eval_text_to_process5.txt" }, 'content': "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."},
    #         {"meta": {"name": "eval_text_to_process55.txt" }, 'content': "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."}
    #     ]
    
    data_dict = {

            # "eval_text_to_process0.txt" : "Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can.",
            # "eval_text_to_process1.txt" : "Neuro-linguistic programming is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s.",
            # "eval_text_to_process2.txt" : "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The goal is a computer capable of \"understanding\" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence. ",
            # "eval_text_to_process3.txt" : "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with computer-human language interactions, specifically how to program computers to process and analyze large amounts of natural language data. The goal is to create a computer that can \"understand\" the contents of documents, including the contextual nuances of the language used in them.The technology can then extract information and insights from the documents, as well as categorize and organize the documents themselves. Natural language processing problems frequently involve speech recognition, natural language comprehension, and natural language generation. Natural language processing dates back to the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" in which he proposed what is now known as the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time was not considered a criterion of intelligence. ",
            "eval_text_to_process4.txt" : "Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that makes human language intelligible to machines. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech.",
            # "eval_text_to_process5.txt" : "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.",
            # "eval_text_to_process55.txt" : "Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding. Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding."
        
    }
    

    haystackapp = Haystack_app()
    
    queries = ["NLP bridges", "NLP", "NLP research", "Natural language processing", "Artificial Intelligence", "language processing"]
    queries = 'what is ' + ' '.join(queries)
    score_dict = haystackapp.evaluator(data_dict, queries)
    print(score_dict)