import wikipediaapi
import validators

class WikiData(object):
    def __init__(self):
        super(WikiData, self).__init__()
        self.wiki_wiki = wikipediaapi.Wikipedia('en')
        self.category_dataURL = []

    def name(self):
        return "WikiData"
    
    def get_wikidata(self, source):
        if type(source) is not str:
            raise NotImplementedError('Implementation error in WikiData input for input type ', type(source) )

        if validators.url(source):
            page_py = self.wiki_wiki.page(title=source.split('/')[-1]) 
        else:
            page_py = self.wiki_wiki.page(title=source)

        return page_py
    
    def get_wikicategorydata(self, subject):
        if type(subject) is not str:
            raise NotImplementedError('Implementation error in WikiData input for input type ', type(subject) )

        return self.wiki_wiki.page(subject)

    def get_wikilinks(self, subject):
        if type(subject) is not str:
            raise NotImplementedError('Implementation error in WikiData input for input type ', type(subject) )

        return self.wiki_wiki.page(subject)

    def print_links(self,page):
            links = page.links
            for title in sorted(links.keys()):
                print("%s: %s" % (title, links[title]))
             
    def print_categorymembers(self, categorymembers, level=0, max_level=1):
            for c in categorymembers.values():
                print("%s: %s (ns: %d)" % ("*" * (level + 1), c.title, c.ns))
                self.category_dataURL.append(c.title)
                self.category_dataURL = list(set(self.category_dataURL))
                if c.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                    self.print_categorymembers(c.categorymembers, level=level + 1, max_level=max_level)
        
if __name__ == "__main__":
    
    wikidata = WikiData()
    page_py = wikidata.get_wikidata('https://en.wikipedia.org/wiki/Natural_language_processing')
    print('=============================================')
    # print("Page - exists(): %s" % page_py.exists())
    # print('----')
    # print("Page - pageid: %s" % page_py.pageid)
    # print('----')
    # print("Page - Title: %s" % page_py.title)
    # print('----')
    # print("Page - summary: %s" % page_py.summary)
    # print('----')
    # print("Page - text: %s" % page_py.text)
    # print('----')
    # print("Page - sections: %s" % page_py.sections)
    # print('----')
    # print("Page - langlinks: %s" % page_py.langlinks)
    # print('----')
    # #     print("Page - section_by_title(name): %s" % page_py.section_by_title(name))
    print('----')
    print("Page - links: %s" % page_py.links)
    print('----')
    # print("Page - categories: %s" % page_py.categories)
    # print('----')
    # print("Page - displaytitle: %s" % page_py.displaytitle)
    # print('----')
    # print("Page - canonicalurl: %s" % page_py.canonicalurl)
    # print('----')
    # print("Page - ns: %s" % page_py.ns)
    # print('----')
    # print("Page - contentmodel: %s" % page_py.contentmodel)
    # print('----')
    # print("Page - pagelanguage: %s" % page_py.pagelanguage)
    # print('----')
    # print("Page - pagelanguagehtmlcode: %s" % page_py.pagelanguagehtmlcode)
    # print('----')
    # print("Page - pagelanguagedir: %s" % page_py.pagelanguagedir)
    # print('----')
    # print("Page - touched: %s" % page_py.touched)
    # print('----')
    # print("Page - lastrevid: %s" % page_py.lastrevid)
    # print('----')
    # print("Page - length: %s" % page_py.length)
    # print('----')
    # print("Page - protection: %s" % page_py.protection)
    # print('----')
    # print("Page - restrictiontypes: %s" % page_py.restrictiontypes)
    # print('----')
    # print("Page - watchers: %s" % page_py.watchers)
    # print('----')
    # print("Page - notificationtimestamp: %s" % page_py.notificationtimestamp)
    # print('----')
    # print("Page - talkid: %s" % page_py.talkid)
    # print('----')
    # print("Page - fullurl: %s" % page_py.fullurl)
    # print('----')
    # print("Page - editurl: %s" % page_py.editurl)
    # print('----')
    # print("Page - readable: %s" % page_py.readable)
    # print('----')
    # print("Page - preload: %s" % page_py.preload)
    
    
    # for title in sorted(page_py.categories.keys()):
    #         print("%s: %s" % (title, page_py.categories[title]))
    #         categorymembers = wikidata.get_wikicategorydata(title)
    #         print(len(categorymembers.categorymembers))
    #         wikidata.print_categorymembers(categorymembers.categorymembers, level=0, max_level=1)
            
    # print('len(wikidata.category_dataURL) : ', len(wikidata.category_dataURL))
    
    wikidata.print_links(page_py)