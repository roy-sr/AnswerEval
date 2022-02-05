from bertopic import BERTopic

class Berttopic(object):
    def __init__(self):
        super(Berttopic, self).__init__()

        self.topic_model = BERTopic(embedding_model="distiluse-base-multilingual-cased-v1", language="english", top_n_words = 10, \
                                        min_topic_size=1, calculate_probabilities = True, verbose=True, n_gram_range=(1, 2))
        
        # tself.opic_model.update_topics()
        
    def name(self):
        return "Berttopic"

    def get_topics(self, docs): 
           
        # self.topics, self.probs = self.topic_model.fit_transform(docs)
        
        return self.topic_model.fit_transform(docs)
        
    def reduce_topics(self, docs, nr_topics):       
        
        if self.topics is None or  self.probs is None:
            self.topics, self.probs = self.topic_model.fit_transform(docs)
            
        # new_topics, new_probs = self.topic_model.reduce_topics(docs, self.topics, self.probs, nr_topics=nr_topics)
        
        return self.topic_model.reduce_topics(docs, self.topics, self.probs, nr_topics=nr_topics)
        
    def get_topic_keywords(self, topic_no):
        
        topic_keywords = []
        for topic_keyword_tuple in self.topic_model.get_topic(topic_no):
            topic_keywords.append(topic_keyword_tuple[0])
            
        return topic_keywords
    

if __name__ == "__main__":
    
    # pip install hdbscan--no - build - isolation--no - binary: all:
    
    eval_text_to_process1 = "Neuro-linguistic programming is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States, in the 1970s."
    eval_text_to_process2 = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.  The goal is a computer capable of \"understanding\" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" which proposed what is now called the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time not articulated as a problem separate from artificial intelligence. "
    eval_text_to_process3 = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with computer-human language interactions, specifically how to program computers to process and analyze large amounts of natural language data. The goal is to create a computer that can \"understand\" the contents of documents, including the contextual nuances of the language used in them.The technology can then extract information and insights from the documents, as well as categorize and organize the documents themselves. Natural language processing problems frequently involve speech recognition, natural language comprehension, and natural language generation. Natural language processing dates back to the 1950s. Already in 1950, Alan Turing published an article titled \"Computing Machinery and Intelligence\" in which he proposed what is now known as the Turing test as a criterion of intelligence, a task that involves the automated interpretation and generation of natural language, but at the time was not considered a criterion of intelligence. "
    eval_text_to_process4 = "Natural Language Processing (NLP) is a field of Artificial Intelligence (AI) that makes human language intelligible to machines. NLP combines the power of linguistics and computer science to study the rules and structure of language, and create intelligent systems (run on machine learning and NLP algorithms) capable of understanding, analyzing, and extracting meaning from text and speech."
    eval_text_to_process5 = "Natural language processing (NLP) is a subfield of Artificial Intelligence (AI). This is a widely used technology for personal assistants that are used in various business fields/areas. This technology works on the speech provided by the user, breaks it down for proper understanding and processes accordingly. This is a very recent and effective approach due to which it has a really high demand in today’s market. Natural Language Processing is an upcoming field where already many transitions such as compatibility with smart devices, interactive talks with a human have been made possible. Knowledge representation, logical reasoning, and constraint satisfaction were the emphasis of AI applications in NLP. Here first it was applied to semantics and later to the grammar. In the last decade, a significant change in NLP research has resulted in the widespread use of statistical approaches such as machine learning and data mining on a massive scale. The need for automation is never ending courtesy of the amount of work required to be done these days. NLP is a very favourable, but aspect when it comes to automated applications. The applications of NLP have led it to be one of the most sought-after methods of implementing machine learning. Natural Language Processing (NLP) is a field that combines computer science, linguistics, and machine learning to study how computers and humans communicate in natural language. The goal of NLP is for computers to be able to interpret and generate human language. This not only improves the efficiency of work done by humans but also helps in interacting with the machine. NLP bridges the gap of interaction between humans and electronic devices."
    
    berttopic = Berttopic()
    
    topics, probs = berttopic.get_topics([eval_text_to_process1, eval_text_to_process2, eval_text_to_process3, eval_text_to_process4, eval_text_to_process5])
    
    print(berttopic.topic_model.get_topic(0))
    