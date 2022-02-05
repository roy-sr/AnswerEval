import logging

class Logger:
    def __init__(self, id):
        # create logger
        self.logger = logging.getLogger(id)
        # self.logger.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.ERROR)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def getLogger(self):
        return self.logger
    
    
#from lib.logger import Logger
# logger.info('Check for status of message')
