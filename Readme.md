1.  Required softaware and libraries:
    ---------------------------------

    A. Stanford CoreNLP docker:
        
        URL: https://github.com/NLPbox/stanford-corenlp-docker


    B. Python lib:

        a. NLTK [url: https://www.nltk.org/]

            i.  stopwords (python -m nltk.downloader stopwords)

            ii. wordnet (python -m nltk.downloader wordnet)

        b.  YAKE [url: https://github.com/LIAAD/yake ]

        c.  Scipy [url: https://scipy.org/]

        d.  Tensorflow Hub [url: https://www.tensorflow.org/hub]

        e.  Google Search Python  [url: https://github.com/Nv7-GitHub/googlesearch]

        f.  WikipediaAPI [url: https://github.com/martin-majlis/Wikipedia-API]

        g.  Validators [url: https://github.com/kvesteri/validators]

        h. Pandas [url: https://pandas.pydata.org/]

        i. Sklearn [url: https://scikit-learn.org/stable/]

        j. Haystack [url: https://github.com/deepset-ai/haystack]


---------------------------------------------------------------------------------------

2.  Execution:
    ----------

    A.  CoreNLP:
        
        docker run -p 9000:9000 corenlp

    B.  Data:

        Input Data in 'dataFile.xlsx'. 

            a.  Sheet 'reference' contains the Question data [Question ID (Unique), Question and Reference Answer]

            b.  Sheet 'candidateAns' contains the responses form the candidates [Answer ID (Unique),	RefQID (reference sheet Question ID), Answer]

    C.  Execute 
        
        Run Demo.ipynb.
        
        For first time it might take time as the libs will download few files in the background (USE, Haystack).

-----------------------------------------------------------------------------------------

3.  References:
    -----------

    a.  Fast and Easy Short Answer Grading with High Accuracy [https://aclanthology.org/N16-1123.pdf]

    b.  Dense Passage Retrieval for Open-Domain Question Answering [https://arxiv.org/pdf/2004.04906.pdf]
