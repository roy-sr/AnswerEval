import tensorflow_hub as hub
try:
    from lib.constants import *
except:
    from constants import *
    
class USE(object):
    def __init__(self):
        self.enc_model_tfr = hub.load(use_url)
        
    def name(self):
        return "USE"

    def get_vectors(self, source):
        
        if type(source) is str:
            source = [source]
        elif type(source) is list:
            pass
        else:
            raise NotImplementedError('Implementation error in USE input for input type ', type(source) )

        return self.enc_model_tfr(source)
    
if __name__ == "__main__":
    use = USE()
    keywords_vectors =use.get_vectors(['hi','hello'])
    import numpy as np
    print(np.array(keywords_vectors).shape)