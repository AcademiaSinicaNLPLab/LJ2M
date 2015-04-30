import sys, os
sys.path.append('../')
from common import utils, filename
from .base import FeatureBase



'''
# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

from features.tfidf import TFIDF

## [dataset, savepath, tfidf_types]

tfidf_obj = TFIDF('LJ2M','/corpus/LJ2M/data/features/tfidf',tf2','idf1','tf3','tf1,'idf2','idf3')
tfidf_obj.calculate('/corpus/LJ2M/data/pkl/lj2m_wordlists')

'''




class TFIDF(FeatureBase):

    class TF:
        def __init__(self, tf_type):
            self.tf_type = tf_type

        def calculate(self, **kwargs):
            if 'Docs_info' in kwargs:
                self.Docs_info = kwargs['Docs_info']
            if 'avg_ld' in kwargs:
                self.avg_ld = kwargs['avg_ld']
            
            if 'tf1' in self.tf_type:
                self._tf1()

            if 'tf2' in self.tf_type:
                self._tf2(1)
            
            if 'tf3' in self.tf_type:
                self._tf3(1, 1)

            return self.Docs_info

        def _tf1(self):
            import numpy as np
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    self.Docs_info[emotion][doc]['tf1'] = {}
                    for word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]

                        tf1 = 1 + np.log2(fd_t)

                        self.Docs_info[emotion][doc]['tf1'][word] = tf1

        def _tf2(self, k):
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    ld = self.Docs_info[emotion][doc]['ld']
                    self.Docs_info[emotion][doc]['tf2'] = {}
                    for word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]

                        tf2 = fd_t / (fd_t + k*(float(ld) / self.avg_ld))

                        self.Docs_info[emotion][doc]['tf2'][word] = tf2

        def _tf3(self, k, b):
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    ld = self.Docs_info[emotion][doc]['ld']
                    self.Docs_info[emotion][doc]['tf3'] = {}
                    for word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]

                        tf3 = (k+1)*fd_t / (fd_t + k*((1-b) + b*(float(ld)/self.avg_ld)))

                        self.Docs_info[emotion][doc]['tf3'][word] = tf3

    class IDF:
        def __init__(self, idf_type):
            self.idf_type = idf_type

        def calculate(self, **kwargs):
            if 'Words_info' in kwargs:
                self.Words_info = kwargs['Words_info']
            if 'D' in kwargs:
                self.D = kwargs['D']
            
            if 'idf1' in self.idf_type:
                self._idf1()

            if 'idf2' in self.idf_type:
                self._idf2()
            
            if 'idf3' in self.idf_type:
                self._idf3()

            return self.Words_info

        def _idf1(self):
            import numpy as np
            for word in self.Words_info:
                ft_D = float(self.Words_info[word]['ft_D'])

                idf1 = np.log(self.D/ft_D)

                self.Words_info[word]['idf1'] = idf1

        def _idf2(self):
            for word in self.Words_info:
                Words_info = dict(self.Words_info)
                del Words_info[word]
                max_nt = max([Words_info[w]['nt'] for w in Words_info])
                nt = self.Words_info[word]['nt']

                idf2 = max_nt - nt

                self.Words_info[word]['idf2'] = idf2

        def _idf3(self):
            import numpy as np
            for word in self.Words_info:
                nt = self.Words_info[word]['nt']

                idf3 = 1 - nt/np.log(self.D)

                self.Words_info[word]['idf3'] = idf3

    def __init__(self, dataset = 'LJ2M', savepath, *tfidf_types):
        self.tf_type = set()
        self.idf_type = set()
        for t in tfidf_types:
            if t.startswith('tf'):
                self.tf_type.add(t)
            elif t.startswith('idf'):
                self.idf_type.add(t)

        self.tf_obj = self.TF(self.tf_type)
        self.idf_obj = self.IDF(self.idf_type)
        self.emotions = filename.emotions[dataset]
        self.savepath = savepath


    def fetch(self, server, collection):
        pass

    def push(self, server, collection):
        pass

    def calculate(self, filename):

        Docs_info = {}         ## documents information contain ld, fd_t, tf
                               ##   of documents in each document, emotion
        Words_info = {}        ## words information contain total_count, ft_D, nt, idf
                               ##   of each word in the whole corpus
        #--------------------------------------------------------------------------       
        T = 0                  ## the universe of terms
        D = 0                  ## the universe of documents
        ft_D = 0               ## the number of documents in D that contain t
        fd_t = 0               ## the number of occurrences of term t in document d
        ld = 0                 ## length of d
        ld_count = 0           ## use for counting avg_ld
        avg_ld = 0             ## average document length in D
        #--------------------------------------------------------------------------

        # self.emotions = self.emotions[33:34]
        for i,emotion in enumerate(self.emotions):
            Docs_info[emotion] = {}
            filepath = os.path.join(filename, emotion+'_wordlists.pkl')
            docs = utils.load_pkl_file(filepath)

            # docs = docs[0:2]
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
        T = len(Words_info)

        Words_info = self.entropy(Docs_info, Words_info)

        Docs_info = self.tf_obj.calculate(Docs_info=Docs_info, avg_ld=avg_ld)
        Words_info = self.idf_obj.calculate(Words_info=Words_info, D=D)

        Docs_info = self.tf_x_idf(Docs_info, Words_info)

        self.dump(Docs_info,self.savepath+'/Docs_info.pkl')
        self.dump(Words_info,self.savepath+'/Words_info.pkl')

    def entropy(self, Docs_info, Words_info):
        import numpy as np
        for word in Words_info:
            nt = 0.
            for emotion in Docs_info:
                for doc in Docs_info[emotion]:
                    if word in Docs_info[emotion][doc]['fd_t']:
                        fd_t = Docs_info[emotion][doc]['fd_t'][word]
                        ft_D = float(Words_info[word]['ft_D'])

                        nt = nt + ((fd_t/ft_D)*np.log(fd_t/ft_D))

            Words_info[word]['nt'] = -nt
        return Words_info

    def tf_x_idf(self, Docs_info, Words_info):
        for tf_type in self.tf_type:
            for idf_type in self.idf_type:
                for emotion in Docs_info:
                    for doc in Docs_info[emotion]:
                        Docs_info[emotion][doc][tf_type+idf_type] = {}
                        for word in Docs_info[emotion][doc][tf_type]:
                            tf = Docs_info[emotion][doc][tf_type][word]
                            idf = Words_info[word][idf_type]

                            tfxidf = tf * idf

                            Docs_info[emotion][doc][tf_type+idf_type][word] = tfxidf
        return Docs_info

    def dump(self,clz,filename):
        utils.save_pkl_file(clz, filename)


    def load(self,filename):
        pass

