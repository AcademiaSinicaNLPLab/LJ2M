import sys, os
sys.path.append('../')
from collections import OrderedDict
from common import utils, filename
from .base import FeatureBase

'''USAGE

# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

from features.tfidf import TFIDF

tfidf_obj = TFIDF('LJ40K','tf3','idf1','tf2')
tfidf_obj.set(tf2_k=1.0, tf3_k=1.0, tf3_b=0.8)
tfidf_obj.calculate('/corpus/LJ40K/data/pkl/lj40k_wordlists')
tfidf_obj.dump('/corpus/LJ40K/data/features/tfidf','tf2k1_tf3k1_tf3b0.8')

'''

class TFIDF(FeatureBase):

    class TF:
        def __init__(self, tf_type):
            self.tf_type = tf_type
            self.tf2_k = 1.0
            self.tf3_k = 1.0
            self.tf3_b = 1.0

        def calculate(self, Docs_info, avg_ld):
            
            self.Docs_info = Docs_info

            for i,emotion in enumerate(self.Docs_info):
                print '(%d/%d) ' % (i+1,len(self.Docs_info)) +emotion+' tf complete!'
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
                            self._tf2(emotion, doc, word, fd_t, ld, avg_ld, k=self.tf2_k)
                        
                        if 'tf3' in self.tf_type:
                            self._tf3(emotion, doc, word, fd_t, ld, avg_ld, k=self.tf3_k, b=self.tf3_b)

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

        def calculate(self, Words_info, D):

            self.Words_info = Words_info

            # calculate max_nt, and what words(max_nt_keys) hold max_nt value.
            if 'idf2' in self.idf_type: 
                max_nt = max([self.Words_info[w]['nt'] for w in self.Words_info])
                max_nt_keys = [key for key,value in self.Words_info.items() if value==max_nt]

            for i,word in enumerate(self.Words_info):
                ft_D = self.Words_info[word]['ft_D']

                if 'idf1' in self.idf_type:
                    self._idf1(word, ft_D, D)

                if 'idf2' in self.idf_type:
                    nt = self.Words_info[word]['nt']
                    # if word has the max_nt value, we need to re-calculate the max_nt except this word.
                    if word in max_nt_keys:
                        Words_info = dict(self.Words_info)
                        del Words_info[word]
                        max_nt = max([Words_info[w]['nt'] for w in Words_info])
                    self._idf2(word, nt, max_nt)
                
                if 'idf3' in self.idf_type:
                    nt = self.Words_info[word]['nt']
                    self._idf3(word, nt, D)

                if (i+1)%50000. == 0.:
                    print '(%d/%d) words idf complete!' % (i+1,len(self.Words_info))
                if (i+1) == len(self.Words_info):
                    print '(%d/%d) words idf complete!' % (i+1,len(self.Words_info))

            return self.Words_info

        def _idf1(self, word, ft_D, D):
            import numpy as np
            #------------------------------------------------------------------#
            idf1 = np.log(float(D)/ft_D)
            #------------------------------------------------------------------#
            self.Words_info[word]['idf1'] = idf1

        def _idf2(self, word, nt, max_nt):
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

        self.Dataset_info = {}              ## dataset information contain: 
                                            ## avg_ld, total_words_count, D, T
        self.Docs_info = OrderedDict()      ## documents information contain: 
                                            ## ld, fd_t, tf
        self.Words_info = {}                ## words information contain:
                                            ## word_total_count, ft_D, nt, idf

        self.emotions = filename.emotions[dataset]

    def set(self, **kwargs):
        if 'tf2' in self.tf_type:
            if 'tf2_k' in kwargs:
                self.tf_obj.tf2_k = kwargs['tf2_k']
        if 'tf3' in self.tf_type:
            if 'tf3_k' in kwargs:
                self.tf_obj.tf3_k = kwargs['tf3_k']
            if 'tf3_b' in kwargs:
                self.tf_obj.tf3_b = kwargs['tf3_b']

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

        print '>> start to preprocessing'
        # self.emotions = self.emotions[98:101]
        for i,emotion in enumerate(self.emotions):
            self.Docs_info[emotion] = {}
            filepath = os.path.join(filename, emotion+'_wordlists.pkl')
            docs = utils.load_pkl_file(filepath)

            # docs = docs[0:10]
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
            print '(%d/%d) ' % (i+1,len(self.emotions)) +emotion+' preprocessing complete!'

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
            print '>> start to calculate tf ', self.tf_type
            ## make tf1, tf2, tf3 (based on your requirement) in self.Docs_info
            self.Docs_info = self.tf_obj.calculate(self.Docs_info, avg_ld)
        

        if len(self.idf_type)>0:
            if 'idf2' in self.idf_type or 'idf3' in self.idf_type:
                print '>> start to calculate entropy'
                ## make nt in self.Words_info
                self.entropy()
            print '>> start to calculate idf ', self.idf_type
            ## make idf1, idf2, idf3 (based on your requirement) in self.Words_info
            self.Words_info = self.idf_obj.calculate(self.Words_info, D)

        if len(self.tf_type)>0 and len(self.idf_type)>0:
            print '>> start to calculate tfidf'
            ## make tfxidf in self.Docs_info (based on what tf, idf pairs in self.Docs_info)
            self.tf_x_idf()

        # print self.Docs_info
        # print '---------------------------------------------------------'
        # print self.Words_info

    def entropy(self):
        import numpy as np
        count = 0
        for i,word in enumerate(self.Words_info):
            print word, ' entropy %d/%d' % (i+1,len(self.Words_info))
            ft_D = float(self.Words_info[word]['ft_D'])
            nt = 0.
            for emotion in self.Docs_info:
                for doc in self.Docs_info[emotion]:
                    if word in self.Docs_info[emotion][doc]['fd_t']:
                        fd_t = self.Docs_info[emotion][doc]['fd_t'][word]
                        #------------------------------------------------------------------#
                        nt = nt + ((fd_t/ft_D)*np.log(fd_t/ft_D))
                        #------------------------------------------------------------------#
            self.Words_info[word]['nt'] = -nt

    def tf_x_idf(self):
        for i,tf_type in enumerate(self.tf_type):
            for j,idf_type in enumerate(self.idf_type):
                print '(%d/%d)  calculating %s x %s' % ((i+1)*(j+1),len(self.tf_type)*len(self.idf_type), tf_type, idf_type)
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

    def dump(self, filepath, postfix):
        print 'start to dump Dataset_info.pkl, Docs_info.pkl, Words_info.pkl in \n'+filepath
        utils.save_pkl_file(self.Docs_info, filepath+'/Docs_info_'+postfix+'.pkl')
        utils.save_pkl_file(self.Words_info, filepath+'/Words_info_'+postfix+'.pkl')
        utils.save_pkl_file(self.Dataset_info, filepath+'/Dataset_info_'+postfix+'.pkl')

    def load(self,filename):
        pass