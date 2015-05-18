# -*- coding: utf8 -*-
import sys, os
from collections import OrderedDict
import numpy as np
from collections import Counter
import cPickle as pickle
from common import utils, filename
from base import FeatureBase
import pymongo
from pymongo import MongoClient
'''USAGE

'''


class TF:
    def __init__(self, tf_type, k=1.0, b=1.0):
        self.tf_type = tf_type
        self.k = k
        self.b = b

    def calculate(self, **kwargs):
        if self.tf_type == 'tf1':
            tf_value = self._tf1(kwargs['fd_t'])
        elif self.tf_type == 'tf2':
            tf_value = self._tf2(kwargs['fd_t'], kwargs['ld'], kwargs['avg_ld'])
        elif self.tf_type == 'tf3':
            tf_value = self._tf3(kwargs['fd_t'], kwargs['ld'], kwargs['avg_ld'])

        return tf_value

    def _tf1(self, fd_t):
        #-------------------------------------------------------------------------------#
        tf1 = 1 + np.log2(float(fd_t))
        #-------------------------------------------------------------------------------#
        return tf1
    def _tf2(self, fd_t, ld, avg_ld):
        #-------------------------------------------------------------------------------#
        tf2 = fd_t / (fd_t + self.k*(float(ld) / avg_ld))
        #-------------------------------------------------------------------------------#
        return tf2
    def _tf3(self, fd_t, ld, avg_ld):
        #-------------------------------------------------------------------------------#
        tf3 = (self.k+1)*fd_t / (fd_t + self.k*((1-self.b) + self.b*(float(ld)/avg_ld)))
        #-------------------------------------------------------------------------------#
        return tf3

class IDF:
    def __init__(self, idf_type):
        self.idf_type = idf_type

    def calculate(self, **kwargs):
        if self.idf_type == 'idf1':
            idf_value = self._idf1(kwargs['ft_D'], kwargs['D'])
        elif self.idf_type == 'idf2':
            idf_value = self._idf2(kwargs['nt'], kwargs['max_nt'])
        elif self.idf_type == 'idf3':
            idf_value = self._idf3(kwargs['nt'], kwargs['D'])

        return idf_value


    def _idf1(self, ft_D, D):
        #------------------------------------------------------------------#
        idf1 = np.log(float(D)/ft_D)
        #------------------------------------------------------------------#
        return idf1
    def _idf2(self, nt, max_nt):
        #------------------------------------------------------------------#
        idf2 = max_nt - nt
        #------------------------------------------------------------------#
        return idf2
    def _idf3(self, nt, D):
        #------------------------------------------------------------------#
        idf3 = 1 - nt/np.log(float(D))
        #------------------------------------------------------------------#
        return idf3

class TFIDF(FeatureBase):
    def __init__(self, tfidf_type, **kwargs):
        tfidf = tfidf_type.split('_')
        b = '' if 'b' not in kwargs else str(kwargs['b']).replace('.','p')
        k = '' if 'k' not in kwargs else str(kwargs['k']).replace('.','p')
        b_postfix = '_b'+b if b != '' else ''
        k_postfix = '_k'+k if k != '' else ''
        postfix = k_postfix+b_postfix
        self.tfidf_type = tfidf[0]+tfidf[1]+postfix


        self.tf_obj = TF(tfidf[0], **kwargs)
        self.idf_obj = IDF(tfidf[1])

    def _calculate_word_level(self, **kwargs):
        tf = self.tf_obj.calculate(fd_t=kwargs['fd_t'], ld=kwargs['ld'], avg_ld=kwargs['DatasetInfo']['avg_ld'])
        idf = self.idf_obj.calculate(ft_D=kwargs['WordInfo']['ft_D'], nt=kwargs['WordInfo']['nt'], max_nt=kwargs['DatasetInfo']['max_nt'], D=kwargs['DatasetInfo']['D'])

        tfidf_value = tf * idf
        return tfidf_value

    def calculate(self, **kwargs):
        DocInfo = kwargs['DocInfo']
        GlobalInfo = kwargs['GlobalInfo']
        DatasetInfo = kwargs['DatasetInfo']
        tfidf_values = []
        for word, fd_t in DocInfo['fd_t']:
            if word in GlobalInfo:
                tfidf_value = self._calculate_word_level(fd_t=fd_t, ld=DocInfo['ld'], WordInfo=GlobalInfo[word], DatasetInfo=DatasetInfo)
                tfidf_values.append([word, tfidf_value])                
        return tfidf_values


    def load(self):
        pass
    def dump(self):
        pass
    def fetch(self):
        pass
    def push(self, db_name, collection_name, emotion, doc_ID, tfidf_values):
        client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
        db = client[db_name]
        collection = db[collection_name]
        collection.update_one({'emotion':emotion,'doc_ID':doc_ID}, {'$set':{self.tfidf_type:tfidf_values}})

class Corpus:
    def __init__(self, dataset, global_loadpath, global_dumppath):
        self.global_loadpath = global_loadpath
        self.global_dumppath = global_dumppath
        self.global_emotions = filename.emotions[dataset]
        self.LocalInfo = []
        self.GlobalInfo = {}
        self.DatasetInfo = {}

    def load_raw_data(self, emotion, filepath):
        loadpath = os.path.join(filepath, emotion+'_wordlists.pkl')
        docs = pickle.load(open(loadpath, 'rb'))
        return docs

    # def load_idx_data(self, filepath):
    #     loadpath = '/home/bs980201/projects/github_repo/LJ40K/batch/random_idx_lj40k_400.pkl'
    #     idxs = pickle.load(open(loadpath, 'rb'))

    def calculate_entropy(self, fd_ts):
        for i, dict_fd_t in enumerate(fd_ts):
            for word in dict_fd_t:
                fd_t = dict_fd_t[word]
                word_total_count = float(self.GlobalInfo[word]['word_total_count'])
                p = fd_t / word_total_count
                #------------------------------------------------------------------#
                nt = p * np.log(p)
                #------------------------------------------------------------------#
                if 'nt' in self.GlobalInfo[word]:
                    self.GlobalInfo[word]['nt'] = self.GlobalInfo[word]['nt'] - nt
                else:
                    self.GlobalInfo[word]['nt'] = -nt

            if (i+1)%10000 == 0:
                print i+1,'/',len(fd_ts)

        print 'calculate max_nt'
        self.DatasetInfo['max_nt'] = max([self.GlobalInfo[w]['nt'] for w in self.GlobalInfo])

    def build_global_info(self):
        #--------------------------------------------------------------------------       
        T = 0                  ## the universe of terms
        D = 0                  ## the universe of documents
        ft_D = 0               ## the number of documents in D that contain t
        fd_ts = []             ## the number of occurrences of term t in document d
        total_words_count = 0  ## use for counting avg_ld
        avg_ld = 0             ## average document length in D
        #--------------------------------------------------------------------------
        if not os.path.exists(self.global_dumppath+'/GlobalInfo.pkl') or not os.path.exists(self.global_dumppath+'/DatasetInfo.pkl'):
            # self.global_emotions = self.global_emotions[0:5]
            for i,emotion in enumerate(self.global_emotions):
                docs = self.load_raw_data(emotion, self.global_loadpath)
                # docs = docs[0:800]
                for d, doc in enumerate(docs):
                    wordlist = sum(doc, [])

                    ld = len(wordlist)
                    total_words_count = total_words_count + ld

                    D = D + 1

                    from collections import Counter
                    dict_fd_t = dict(Counter(wordlist))
                    ## e.g. dict(fd_t) = {'happy' : 3, 'to' : 10, 'code' : 5}
                    fd_ts.append(dict_fd_t)

                    for word in dict_fd_t.keys():
                        if word in self.GlobalInfo:
                            self.GlobalInfo[word]['word_total_count'] = self.GlobalInfo[word]['word_total_count'] + dict_fd_t[word]
                            self.GlobalInfo[word]['ft_D'] = self.GlobalInfo[word]['ft_D'] + 1
                        else:
                            self.GlobalInfo[word] = {}
                            self.GlobalInfo[word]['word_total_count'] = dict_fd_t[word]
                            self.GlobalInfo[word]['ft_D'] = 1
                del docs    
                print '(%d/%d) ' % (i+1,len(self.global_emotions)) +emotion+' Global preprocessing complete!'

            avg_ld = total_words_count / D
            T = len(self.GlobalInfo)

            print 'total_words_count: ', total_words_count
            print 'avg_ld: ', avg_ld
            print 'D: ', D
            print 'T: ', T

            self.DatasetInfo['total_words_count'] = total_words_count
            self.DatasetInfo['avg_ld'] = avg_ld
            self.DatasetInfo['D'] = D
            self.DatasetInfo['T'] = T

            self.calculate_entropy(fd_ts)
        else:
            print 'loading GlobalInfo.pkl and DatasetInfo.pkl'
            self.GlobalInfo = pickle.load(open(self.global_dumppath+'/GlobalInfo.pkl','rb'))
            self.DatasetInfo = pickle.load(open(self.global_dumppath+'/DatasetInfo.pkl','rb'))

    def build_local_info(self, dataset, local_loadpath, db_name, collection_name):
        '''
        deal with training data and testing data
        '''
        self.local_emotions = filename.emotions[dataset]

        client = MongoClient('doraemon.iis.sinica.edu.tw:27017')
        db = client[db_name]
        if collection_name not in db.collection_names():
            push = True
            self.collection = db[collection_name]
        else: 
            push = False

        # self.local_emotions = self.local_emotions[0:2]
        for e_ID, emotion in enumerate(self.local_emotions):
            docs = self.load_raw_data(emotion, local_loadpath)
            emotion_info = []
            # docs = docs[0:5]
            for doc_ID, doc in enumerate(docs):
                wordlist = sum(doc, [])

                doc_info = {}

                ld = len(wordlist)
                doc_info['ld'] = ld

                fd_ts = Counter(wordlist)
                fd_ts = sorted(fd_ts.items())
                ## e.g. fd_ts.items() = [('happy',3), ('to',10), ('code',5)]
                doc_info['fd_t'] = fd_ts
                emotion_info.append(doc_info)
                if push:
                    self.push_local_info(emotion, e_ID, doc_ID, ld, fd_ts)
            
            self.LocalInfo.append(emotion_info)
            print '(%d/%d) ' % (e_ID+1,len(self.local_emotions)) +emotion+' Local preprocessing complete!'


    def push_local_info(self, emotion, e_ID, doc_ID, ld, fd_ts):
        doc_info = {}
        doc_info['emotion'] = emotion
        doc_info['e_ID'] = e_ID
        doc_info['doc_ID'] = doc_ID
        doc_info['ld'] = ld
        doc_info['fd_t'] = fd_ts
        self.collection.insert_one(doc_info)

    def update_local_info(self, e_ID, doc_ID, tfidf_type, tfidf_values):
        self.LocalInfo[e_ID][doc_ID][tfidf_type] = tfidf_values

    def get_local_info(self):
        return self.LocalInfo

    def get_global_info(self):
        return self.GlobalInfo

    def get_dataset_info(self):
        return self.DatasetInfo

    def dump_local(self, e_ID, save_path):
        utils.save_pkl_file(self.LocalInfo[e_ID], save_path)

    def dump_global(self):
        utils.save_pkl_file(self.GlobalInfo, self.global_dumppath+'/GlobalInfo.pkl')
        utils.save_pkl_file(self.DatasetInfo, self.global_dumppath+'/DatasetInfo.pkl')


##-------------------------------------------------------------------------------------------------------------##
# -*- coding: utf8 -*-
import sys, os
sys.path.append("../")

# from features.tfidftest3 import TFIDF
global_dataset_name = 'LJ2M'
local_dataset_name = 'LJ2M'
db_name = 'LJ2M'
collection_name = 'TFIDF_feature'
global_dataset_load_path = '/corpus/LJ2M/data/pkl/lj2m_wordlists'
local_dataset_load_path = '/corpus/LJ2M/data/pkl/lj2m_wordlists'


global_dumppath = '/corpus/LJ2M/data/features/tfidf'
local_dumppath = '/corpus/LJ2M/data/features/tfidf/LocalInfo'
if local_dumppath and not os.path.exists(local_dumppath): os.makedirs(local_dumppath)


c = Corpus(global_dataset_name, global_dataset_load_path, global_dumppath)
c.build_global_info()
c.dump_global()
c.build_local_info(local_dataset_name, local_dataset_load_path , db_name, collection_name)

TFIDF_obj = TFIDF('tf3_idf2', k=1.0, b=0.8)
LocalInfo = c.get_local_info()
GlobalInfo = c.get_global_info()
DatasetInfo = c.get_dataset_info()

for e_ID, emotion_docs in enumerate(LocalInfo):
    emotion = c.local_emotions[e_ID]
    for doc_ID, docs in enumerate(emotion_docs):
        tfidf_values = TFIDF_obj.calculate(DocInfo=docs, GlobalInfo=GlobalInfo, DatasetInfo=DatasetInfo) 
        TFIDF_obj.push(db_name, collection_name, emotion, doc_ID, tfidf_values)
        # c.update_local_info(e_ID, doc_ID, TFIDF_obj.tfidf_type, tfidf_values)

    # save_path = os.path.join(local_dumppath, emotion+'_LocalInfo.pkl')
    # c.dump_local(e_ID, save_path)

    print '(%d/%d) ' % (e_ID+1,len(LocalInfo)) +emotion+' TFIDF complete!'


## 1. do npz pkl features, (load from mongo and filter <10), sentence level
## 3. support idx
## 4. 
