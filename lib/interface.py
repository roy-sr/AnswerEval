import requests
import json


try:
    from lib.constants import *
except:
    from constants import *




class ApiInterface:
    
    def __init__(self):
        super(ApiInterface, self).__init__()

    
    
    def get_stanfordNLU_data(self,text, properties=None ):

        data = data = text.encode()

        resp = requests.post(stanfordNLU_server_url, params={
                                "properties": json.dumps(properties)
                            }, data=data, headers={'Connection': 'close'})
        
        return json.loads(resp.text)
    
    
    def get_google_cookies(self):

        response = requests.get('https://google.com')
        if response.status_code == 200:
            return response.cookies.get_dict()
        return None
            
            
            
if __name__ == "__main__":

    interfaceUtils = ApiInterface()
    out =  interfaceUtils.get_stanfordNLU_data(' Hello',properties = {"annotators": "tokenize,ssplit,pos,lemma", "outputFormat": "json"})
    print(out)
    print(len(out['sentences']))
    
    """
    {'sentences': []}
    
    {'sentences': 
        [
            {'index': 0, 
            'tokens': 
                [
                    {'index': 1, 'word': 'jumped', 'originalText': 'jumped', 'lemma': 'jump', 'characterOffsetBegin': 0, 'characterOffsetEnd': 6, 'pos': 'VBD', 'before': '', 'after': ''}
                ]
            }
        ]
    }
    
    {'sentences': 
        [
            {'index': 0, 
            'tokens': 
                [
                    {'index': 1, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 0, 'characterOffsetEnd': 3, 'pos': 'DT', 'before': '', 'after': ' '}, 
                    {'index': 2, 'word': 'quick', 'originalText': 'quick', 'lemma': 'quick', 'characterOffsetBegin': 4, 'characterOffsetEnd': 9, 'pos': 'JJ', 'before': ' ', 'after': ' '}, 
                    {'index': 3, 'word': 'brown', 'originalText': 'brown', 'lemma': 'brown', 'characterOffsetBegin': 10, 'characterOffsetEnd': 15, 'pos': 'JJ', 'before': ' ', 'after': ' '}, 
                    {'index': 4, 'word': 'fox', 'originalText': 'fox', 'lemma': 'fox', 'characterOffsetBegin': 16, 'characterOffsetEnd': 19, 'pos': 'NN', 'before': ' ', 'after': ' '}, 
                    {'index': 5, 'word': 'jumped', 'originalText': 'jumped', 'lemma': 'jump', 'characterOffsetBegin': 20, 'characterOffsetEnd': 26, 'pos': 'VBD', 'before': ' ', 'after': ' '}, 
                    {'index': 6, 'word': 'over', 'originalText': 'over', 'lemma': 'over', 'characterOffsetBegin': 27, 'characterOffsetEnd': 31, 'pos': 'IN', 'before': ' ', 'after': ' '}, 
                    {'index': 7, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 32, 'characterOffsetEnd': 35, 'pos': 'DT', 'before': ' ', 'after': ' '}, 
                    {'index': 8, 'word': 'lazy', 'originalText': 'lazy', 'lemma': 'lazy', 'characterOffsetBegin': 36, 'characterOffsetEnd': 40, 'pos': 'JJ', 'before': ' ', 'after': ' '}, 
                    {'index': 9, 'word': 'dog', 'originalText': 'dog', 'lemma': 'dog', 'characterOffsetBegin': 41, 'characterOffsetEnd': 44, 'pos': 'NN', 'before': ' ', 'after': ''}
                ]
            }
        ]
    }
    
    {'sentences': 
        [
            {'index': 0, 
            'tokens': 
                [
                    {'index': 1, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 0, 'characterOffsetEnd': 3, 'pos': 'DT', 'before': '', 'after': ' '},
                    {'index': 2, 'word': 'quick', 'originalText': 'quick', 'lemma': 'quick', 'characterOffsetBegin': 4, 'characterOffsetEnd': 9, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 3, 'word': 'brown', 'originalText': 'brown', 'lemma': 'brown', 'characterOffsetBegin': 10, 'characterOffsetEnd': 15, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 4, 'word': 'fox', 'originalText': 'fox', 'lemma': 'fox', 'characterOffsetBegin': 16, 'characterOffsetEnd': 19, 'pos': 'NN', 'before': ' ', 'after': ' '},
                    {'index': 5, 'word': 'jumped', 'originalText': 'jumped', 'lemma': 'jump', 'characterOffsetBegin': 20, 'characterOffsetEnd': 26, 'pos': 'VBD', 'before': ' ', 'after': ' '},
                    {'index': 6, 'word': 'over', 'originalText': 'over', 'lemma': 'over', 'characterOffsetBegin': 27, 'characterOffsetEnd': 31, 'pos': 'IN', 'before': ' ', 'after': ' '},
                    {'index': 7, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 32, 'characterOffsetEnd': 35, 'pos': 'DT', 'before': ' ', 'after': ' '},
                    {'index': 8, 'word': 'lazy', 'originalText': 'lazy', 'lemma': 'lazy', 'characterOffsetBegin': 36, 'characterOffsetEnd': 40, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 9, 'word': 'dog', 'originalText': 'dog', 'lemma': 'dog', 'characterOffsetBegin': 41, 'characterOffsetEnd': 44, 'pos': 'NN', 'before': ' ', 'after': ''},
                    {'index': 10, 'word': '.', 'originalText': '.', 'lemma': '.', 'characterOffsetBegin': 44, 'characterOffsetEnd': 45, 'pos': '.', 'before': '', 'after': ' '}
                ]
            },
            {'index': 1, 
            'tokens': 
                [
                    {'index': 1, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 46, 'characterOffsetEnd': 49, 'pos': 'DT', 'before': ' ', 'after': ' '},
                    {'index': 2, 'word': 'quick', 'originalText': 'quick', 'lemma': 'quick', 'characterOffsetBegin': 50, 'characterOffsetEnd': 55, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 3, 'word': 'brown', 'originalText': 'brown', 'lemma': 'brown', 'characterOffsetBegin': 56, 'characterOffsetEnd': 61, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 4, 'word': 'fox', 'originalText': 'fox', 'lemma': 'fox', 'characterOffsetBegin': 62, 'characterOffsetEnd': 65, 'pos': 'NN', 'before': ' ', 'after': ' '},
                    {'index': 5, 'word': 'jumped', 'originalText': 'jumped', 'lemma': 'jump', 'characterOffsetBegin': 66, 'characterOffsetEnd': 72, 'pos': 'VBD', 'before': ' ', 'after': ' '},
                    {'index': 6, 'word': 'over', 'originalText': 'over', 'lemma': 'over', 'characterOffsetBegin': 73, 'characterOffsetEnd': 77, 'pos': 'IN', 'before': ' ', 'after': ' '},
                    {'index': 7, 'word': 'the', 'originalText': 'the', 'lemma': 'the', 'characterOffsetBegin': 78, 'characterOffsetEnd': 81, 'pos': 'DT', 'before': ' ', 'after': ' '},
                    {'index': 8, 'word': 'lazy', 'originalText': 'lazy', 'lemma': 'lazy', 'characterOffsetBegin': 82, 'characterOffsetEnd': 86, 'pos': 'JJ', 'before': ' ', 'after': ' '},
                    {'index': 9, 'word': 'dog', 'originalText': 'dog', 'lemma': 'dog', 'characterOffsetBegin': 87, 'characterOffsetEnd': 90, 'pos': 'NN', 'before': ' ', 'after': ''},
                    {'index': 10, 'word': '.', 'originalText': '.', 'lemma': '.', 'characterOffsetBegin': 90, 'characterOffsetEnd': 91, 'pos': '.', 'before': '', 'after': ' '}
                ]
            }
        ]
    }
    """