import sys, os
sys.path.append('../')
from common import utils, filename
from .base import FeatureBase

class TFIDF(FeatureBase):

    class TF:
        def __init__(self, tf_type):
            self.tf_type = tf_type

        def _tf1(self):
            pass

        def _tf2(self, k):
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    ld = self.Docs_info[emotion][doc]['ld']
                    self.Docs_info[emotion][doc]['tf2'] = {}
                    for word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]

                        tf2 = fd_t / (fd_t + k*(ld / self.avg_ld))
                        self.Docs_info[emotion][doc]['tf2'][word] = tf2

            print self.Docs_info

        def _tf3(self):
            pass

        def calculate(self, **kwargs):
            if 'Docs_info' in kwargs:
                self.Docs_info = kwargs['Docs_info']
            if 'Words_info' in kwargs:
                self.Words_info = kwargs['Words_info']
            if 'D' in kwargs:
                self.D = kwargs['D']
            if 'avg_ld' in kwargs:
                self.avg_ld = kwargs['avg_ld']
            
            if self.tf_type == 'tf1':
                self._tf1()

            elif self.tf_type == 'tf2':
                self._tf2(1)
            
            elif self.tf_type == 'tf3':
                self._tf3()

    class IDF:
        def __init__(self, idf_type):
            self.idf_type = idf_type

    def __init__(self, dataset = 'LJ2M', tf_type = 'tf2', idf_type = 'idf2'):
        self.tf_obj = self.TF(tf_type)
        self.idf_obj = self.IDF(idf_type)
        self.emotions = filename.emotions[dataset]


    def fetch(self, server, collection):
        pass

    def push(self, server, collection):
        pass

    def calculate(self, filename):

        Docs_info = {}         ## documents information contain ld, fd_t 
                               ##   of documents in each document, emotion
        Words_info = {}        ## words information contain total_count, ft_D
                               ##   of each word in the whole corpus
        #--------------------------------------------------------------------------       
        T = 0                  ## the universe of terms
        T_set = set()          ## the set of all terms
        D = 0                  ## the universe of documents
        ft_D = 0               ## the number of documents in D that contain t
        fd_t = 0               ## the number of occurrences of term t in document d
        ld = 0                 ## length of d
        ld_count = 0           ## use for counting avg_ld
        avg_ld = 0             ## average document length in D
        #--------------------------------------------------------------------------

        self.emotions = self.emotions[33:34]
        for i,emotion in enumerate(self.emotions):
            Docs_info[emotion] = {}
            filepath = os.path.join(filename, emotion+'_wordlists.pkl')
            docs = utils.load_pkl_file(filepath)

            docs = docs[0:2]
            for d, doc in enumerate(docs):
                wordlist = sum(doc, [])
                Docs_info[emotion][d] = {}

                ld = len(wordlist)
                Docs_info[emotion][d]['ld'] = ld
                ld_count = ld_count + ld

                from collections import Counter
                fd_t = Counter(wordlist)
                ## e.g. dict(fd_t) = {'happy' : 3, 'to' : 10, 'code' : 5}
                Docs_info[emotion][d]['fd_t'] = dict(fd_t)

                for word in dict(fd_t).keys():
                    if word in Words_info:
                        Words_info[word]['total_count'] = Words_info[word]['total_count'] + dict(fd_t)[word]
                        Words_info[word]['ft_D'] = Words_info[word]['ft_D'] + 1
                    else:
                        Words_info[word] = {}
                        Words_info[word]['total_count'] = dict(fd_t)[word]
                        Words_info[word]['ft_D'] = 1
                D = D + 1

        avg_ld = ld_count / D

        T_set = set(Words_info.keys())
        T = len(Words_info)

        # print T_set
        # print T
        # print D
        # print avg_ld
        # print Docs_info
        # print Words_info

        self.tf_obj.calculate(Docs_info=Docs_info, Words_info=Words_info, D=D, avg_ld=avg_ld)

    def dump(self,filename):
        pass


    def load(self,filename):
        pass

