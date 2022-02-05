from collections import *

from googlesearch import search

# to search
# site:wikipedia.org natural language generation
# qword = 'natural language generation'
# query = "site:wikipedia.org " + qword

class QueryGoogle(object):
    def __init__(self, tld='com', lang='en', tbs='0', safe='off', num=10, start=0, stop=10, pause=2.0, country='', \
                        extra_params=None, user_agent=None, verify_ssl=True):
        super(QueryGoogle, self).__init__()
        """
        tld='com', lang='en', tbs='0', safe='off', num=10, start=0,
                stop=None, pause=2.0, country='', extra_params=None,
                user_agent=None, verify_ssl=True
            Search the given query string using Google.

        :param str query: Query string. Must NOT be url-encoded.
        :param str tld: Top level domain.
        :param str lang: Language.
        :param str tbs: Time limits (i.e "qdr:h" => last hour,
            "qdr:d" => last 24 hours, "qdr:m" => last month).
        :param str safe: Safe search.
        :param int num: Number of results per page.
        :param int start: First result to retrieve.
        :param int stop: Last result to retrieve.
            Use None to keep searching forever.
        :param float pause: Lapse to wait between HTTP requests.
            A lapse too long will make the search slow, but a lapse too short may
            cause Google to block your IP. Your mileage may vary!
        :param str country: Country or region to focus the search on. Similar to
            changing the TLD, but does not yield exactly the same results.
            Only Google knows why...
        :param dict extra_params: A dictionary of extra HTTP GET
            parameters, which must be URL encoded. For example if you don't want
            Google to filter similar results you can set the extra_params to
            {'filter': '0'} which will append '&filter=0' to every query.
        :param str user_agent: User agent for the HTTP requests.
            Use None for the default.
        :param bool verify_ssl: Verify the SSL certificate to prevent
            traffic interception attacks. Defaults to True.

        :rtype: generator of str
        :return: Generator (iterator) that yields found URLs.
            If the stop parameter is None the iterator will loop forever.
        """
        
        self.tld=tld
        self.lang=lang
        self.tbs=tbs
        self.safe=safe
        self.num=num
        self.start=start
        self.stop=stop
        self.pause=pause
        self.country=country
        self.extra_params=extra_params
        self.user_agent=user_agent
        self.verify_ssl=verify_ssl
        
        self.query = ''
        self.source = ''
        
    def name(self):
        return "QueryGoogle"
    
    def set_source(self, source = None):
        
        if source != None:
            self.query = "site:" + source + ' '

    def search_google(self, query):
        return search(query, tld=self.tld, lang=self.lang, tbs=self.tbs, safe=self.safe, num=self.num, start=self.start,
                        stop=self.stop, pause=self.pause, country=self.country, extra_params=self.extra_params,
                            user_agent=self.user_agent, verify_ssl=self.verify_ssl)

    def search_google_unique_URL(self, query):
        
        url_list = []
        for j in search(self.query + query, tld=self.tld, lang=self.lang, tbs=self.tbs, safe=self.safe, num=self.num, start=self.start,
                        stop=self.stop*4, pause=self.pause, country=self.country, extra_params=self.extra_params,
                            user_agent=self.user_agent, verify_ssl=self.verify_ssl):
            
            page_url = j.split('#')[0]
            if page_url not in url_list:
                url_list.append(j.split('#')[0])
                
            if len(url_list) > self.stop:
                break
            
        return url_list

if __name__ == "__main__":
    
    querygoogle = QueryGoogle()
    querygoogle.set_source('wikipedia.org')
    for j in querygoogle.search_google_unique_URL('NLP'):
        print(j)