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
                avg_ld = kwargs['avg_ld']

            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    ld = self.Docs_info[emotion][doc]['ld']

                    if 'tf1' in self.tf_type:
                        self.Docs_info[emotion][doc]['tf1'] = {}
                    if 'tf2' in self.tf_type:
                        self.Docs_info[emotion][doc]['tf2'] = {}
                    if 'tf3' in self.tf_type:
                        self.Docs_info[emotion][doc]['tf3'] = {}

                    for word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]

                        if 'tf1' in self.tf_type:
                            self._tf1(emotion, doc, word, fd_t)

                        if 'tf2' in self.tf_type:
                            self._tf2(emotion, doc, word, fd_t, ld, avg_ld, k=1.)
                        
                        if 'tf3' in self.tf_type:
                            self._tf3(emotion, doc, word, fd_t, ld, avg_ld, k=1., b=1.)

            return self.Docs_info

        def _tf1(self, emotion, doc, word, fd_t):
            import numpy as np
            #------------------------------------------------------------------#
            tf1 = 1 + np.log2(float(fd_t))
            #------------------------------------------------------------------#
            self.Docs_info[emotion][doc]['tf1'][word] = tf1

        def _tf2(self, emotion, doc, word, fd_t, ld, avg_ld, k=1.):
            #------------------------------------------------------------------#
            tf2 = fd_t / (fd_t + k*(float(ld) / avg_ld))
            #------------------------------------------------------------------#
            self.Docs_info[emotion][doc]['tf2'][word] = tf2

        def _tf3(self, emotion, doc, word, fd_t, ld, avg_ld, k=1., b=1.):
            #------------------------------------------------------------------#
            tf3 = (k+1)*fd_t / (fd_t + k*((1-b) + b*(float(ld)/avg_ld)))
            #------------------------------------------------------------------#
            self.Docs_info[emotion][doc]['tf3'][word] = tf3

    class IDF:
        def __init__(self, idf_type):
            self.idf_type = idf_type

        def calculate(self, **kwargs):
            if 'Words_info' in kwargs:
                self.Words_info = kwargs['Words_info']
            if 'D' in kwargs:
                D = kwargs['D']
            
            for word in self.Words_info:
                ft_D = self.Words_info[word]['ft_D']
                nt = self.Words_info[word]['nt']

                if 'idf1' in self.idf_type:
                    self._idf1(word, ft_D, D)

                if 'idf2' in self.idf_type:
                    self._idf2(word, nt)
                
                if 'idf3' in self.idf_type:
                    self._idf3(word, nt, D)

            return self.Words_info

        def _idf1(self, word, ft_D, D):
            import numpy as np
            #------------------------------------------------------------------#
            idf1 = np.log(float(D)/ft_D)
            #------------------------------------------------------------------#
            self.Words_info[word]['idf1'] = idf1

        def _idf2(self, word, nt):
            Words_info = dict(self.Words_info)
            del Words_info[word]
            max_nt = max([Words_info[w]['nt'] for w in Words_info])
            #------------------------------------------------------------------#
            idf2 = max_nt - nt
            #------------------------------------------------------------------#
            self.Words_info[word]['idf2'] = idf2

        def _idf3(self, word, nt, D):
            import numpy as np
            #------------------------------------------------------------------#
            idf3 = 1 - nt/np.log(float(D))
            #------------------------------------------------------------------#
            self.Words_info[word]['idf3'] = idf3

    def __init__(self, dataset = 'LJ2M', *tfidf_types):
        self.tf_type = set()
        self.idf_type = set()
        for t in tfidf_types:
            if t.startswith('tf'):
                self.tf_type.add(t)
            elif t.startswith('idf'):
                self.idf_type.add(t)
        self.tf_obj = self.TF(self.tf_type)
        self.idf_obj = self.IDF(self.idf_type)

        self.Dataset_info = {}      ## dataset information contain: 
                                    ## avg_ld, total_words_count, D, T
        self.Docs_info = {}         ## documents information contain: 
                                    ## ld, fd_t, tf
        self.Words_info = {}        ## words information contain:
                                    ## word_total_count, ft_D, nt, idf

        self.emotions = filename.emotions[dataset]

    def fetch(self, server, collection):
        pass

    def push(self, server, collection):
        pass

    def calculate(self, filename):
        #--------------------------------------------------------------------------       
        T = 0                  ## the universe of terms
        D = 0                  ## the universe of documents
        ft_D = 0               ## the number of documents in D that contain t
        fd_t = 0               ## the number of occurrences of term t in document d
        ld = 0                 ## length of d
        total_words_count = 0  ## use for counting avg_ld
        avg_ld = 0             ## average document length in D
        #--------------------------------------------------------------------------

        print 'start to preprocessing'
        self.emotions = self.emotions[33:35]
        for i,emotion in enumerate(self.emotions):
            self.Docs_info[emotion] = {}
            filepath = os.path.join(filename, emotion+'_wordlists.pkl')
            docs = utils.load_pkl_file(filepath)

            docs = docs[0:5]
            for d, doc in enumerate(docs):
                wordlist = sum(doc, [])
                self.Docs_info[emotion][d] = {}

                ld = len(wordlist)
                self.Docs_info[emotion][d]['ld'] = ld
                total_words_count = total_words_count + ld

                from collections import Counter
                fd_t = Counter(wordlist)
                ## e.g. dict(fd_t) = {'happy' : 3, 'to' : 10, 'code' : 5}
                self.Docs_info[emotion][d]['fd_t'] = dict(fd_t)

                for word in dict(fd_t).keys():
                    if word in self.Words_info:
                        self.Words_info[word]['word_total_count'] = self.Words_info[word]['word_total_count'] + dict(fd_t)[word]
                        self.Words_info[word]['ft_D'] = self.Words_info[word]['ft_D'] + 1
                    else:
                        self.Words_info[word] = {}
                        self.Words_info[word]['word_total_count'] = dict(fd_t)[word]
                        self.Words_info[word]['ft_D'] = 1
                D = D + 1
            print emotion+'(%d/%d) preprocessing complete!' % (i,len(self.emotions))

        avg_ld = total_words_count / D
        T = len(self.Words_info)

        self.Dataset_info['total_words_count'] = total_words_count
        self.Dataset_info['avg_ld'] = avg_ld
        self.Dataset_info['D'] = D
        self.Dataset_info['T'] = T

        ## already get T, D, ft_D, fd_t, ld, total_words_count, avg_ld, word_total_count
        ## in three dict: self.Dataset_info, self.Docs_info, self.Words_info
        ##------------------------------------------------------------------------------##
        ## start to use this parameters to yield nt                  in self.Words_info 
        ##                                       tf, idf, tfidf      in self.Docs_info

        if len(self.tf_type)>0:
            print 'start to calculate tf'
            ## make tf1, tf2, tf3 (based on your requirement) in self.Docs_info
            self.Docs_info = self.tf_obj.calculate(Docs_info=self.Docs_info, avg_ld=avg_ld)

        ## make nt in self.Words_info
        self.entropy()

        if len(self.idf_type)>0:
            print 'start to calculate idf'
            ## make idf1, idf2, idf3 (based on your requirement) in self.Words_info
            self.Words_info = self.idf_obj.calculate(Words_info=self.Words_info, D=D)

        if len(self.tf_type)>0 and len(self.idf_type)>0:
            print 'start to calculate tfidf'
            ## make tfxidf in self.Docs_info (based on what tf, idf pairs in self.Docs_info)
            self.tf_x_idf()

        # print self.Docs_info
        # print '---------------------------------------------------------'
        # print self.Words_info

    def entropy(self):
        import numpy as np
        for word in self.Words_info:
            nt = 0.
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    if word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]
                        ft_D = float(self.Words_info[word]['ft_D'])
                        #------------------------------------------------------------------#
                        nt = nt + ((fd_t/ft_D)*np.log(fd_t/ft_D))
                        #------------------------------------------------------------------#
            self.Words_info[word]['nt'] = -nt

    def tf_x_idf(self):
        for tf_type in self.tf_type:
            for idf_type in self.idf_type:
                for emotion in self.Docs_info:
                    for doc in self.Docs_info[emotion]:
                        self.Docs_info[emotion][doc][tf_type+idf_type] = {}
                        for word in self.Docs_info[emotion][doc][tf_type]:
                            tf = self.Docs_info[emotion][doc][tf_type][word]
                            idf = self.Words_info[word][idf_type]
                            #------------------------------------------------------------------#
                            tfxidf = tf * idf
                            #------------------------------------------------------------------#
                            self.Docs_info[emotion][doc][tf_type+idf_type][word] = tfxidf

    def dump(self, filepath):
        print 'start to dump Dataset_info.pkl, Docs_info.pkl, Words_info.pkl in \n'+filepath
        utils.save_pkl_file(self.Docs_info, filepath+'/Docs_info.pkl')
        utils.save_pkl_file(self.Words_info, filepath+'/Words_info.pkl')
        utils.save_pkl_file(self.Dataset_info, filepath+'/Dataset_info.pkl')

    def load(self,filename):
        pass