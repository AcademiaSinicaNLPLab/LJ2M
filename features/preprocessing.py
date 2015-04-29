
import os
import sys
sys.path.append('../')

import logging
import random
import pickle
import json
import numpy as np

from common import filename
from common import utils


class FeatureList:
    """
    """
    def __init__(self, filename):

        try:
            fp = open(filename, 'r')
            self.feature_list = json.load(fp)
            fp.close()
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except ValueError:
            print "Could not convert data to an integer."
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    def __iter__(self):
        """
            iterate on the feature name
        """
        for feature in self.feature_list:
            yield (feature['name'], feature['data'])            

    @staticmethod
    def get_full_data_path(emotion, data):
        fname = '.'.join([data['file_prefix'], emotion, data['file_postfix']])
        return os.path.join(data['dir'], fname)

    def _get_file_name_by_emtion(self, data_dir, emotion, **kwargs):
        '''
            serach the data_dir and get the file name with the specified emotion and extension
        '''
        ext = '.pkl' if 'ext' not in kwargs else kwargs['ext']
        files = [fname for fname in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, fname))]

        # target file is the file that contains the emotion string and has the desginated extension
        for fname in files:
            target = None
            if fname.endswith(ext) and fname.find(emotion) != -1:
                target = fname
                break
        return target

    def get_paths_by_emotion(self, emotion_name):
        paths = []
        for feature in self.feature_list:         
            fname = self._get_file_name_by_emtion(feature['data_dir'], emotion_name, exp='.pkl')
            if fname is not None:
                paths.append(os.path.join(feature['data_dir'], fname))
            else:
                raise ValueError("failed to find the data of %s in %s" % (emotion_name, feature['data_dir']))
        return paths


class Dataset:
    """
        store train/dev/test datasets
    """
    def __init__(self, data_path, **kwargs):
        """
            options:
                loglevel
        """
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        # only use 40 emotions
        self.emotions = filename.emotions['LJ40K']
        self.X = {}

        for emotion in self.emotions:

            fpath = FeatureList.get_full_data_path(emotion, data_path)
            self.logger.info("load features from %s", fpath)
            try:
                Xy = pickle.load(open(fpath, "r"))
            except ValueError:
                self.logger.error("failed to load %s" % (fpath))

            self.X[emotion] = np.zeros((len(Xy), 300), dtype="float32")
            for i in range(len(Xy)):    
                # make sure only one feature vector in each doc
                assert Xy[i]['X'].shape[0] == 1         
                self.X[emotion][i] = Xy[i]['X']

    def get_dataset(self, emotion, idxs, set_type):

        idxs_typed = idxs[set_type]

        n_pos = len(idxs_typed[emotion][emotion])
        n_neg = (n_pos / 39) * 39   # hidden rule

        y = ([1]*n_pos) + ([-1]*n_neg)

        # to save time we allocate the space first
        X = np.zeros((n_pos+n_neg, 300), dtype="float32")

        X[:n_pos] = self.X[emotion][idxs_typed[emotion][emotion]]
        cnt = n_pos
        
        # process negatives
        for neg_emotion in filter(lambda x: x != emotion, self.emotions):

            idxs_neg = idxs_typed[emotion][neg_emotion]

            n_neg_typed = len(idxs_neg)
            assert n_neg_typed == n_neg/39
            
            assert X[cnt:cnt+n_neg_typed].shape == self.X[neg_emotion][idxs_neg].shape
            X[cnt:cnt+n_neg_typed] = self.X[neg_emotion][idxs_neg]
            cnt += n_neg_typed
        
        assert cnt == n_pos + n_neg
        return X, y


class FusedDataset:
    """
        fuse different features
    """
    def __init__(self, idxs, **kwargs):
        """
            idxs['train'][emotion][emotion]
            idxs['dev'][emotion][emotion]
            idxs['test'][emotion][emotion]

            options:
                loglevel
        """

        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        self.idxs = idxs
        self.feature_name = []
        self.datasets = []

    def add_feature(self, feature_name, dataset):
        self.feature_name.append(feature_name)
        self.datasets.append(dataset)

    def get_feature_name(self):
        '+'.join(self.feature_name)

    def get_dataset(self, emotion, set_type):

        X = None
        y = None
        
        for dataset in self.datasets:
            X_temp, y_temp = dataset.get_dataset(emotion, self.idxs, set_type)

            if y == None:
                y = y_temp
                X = X_temp
            else:   # ToDo: not test yet
                assert (y == y_temp)
                X = np.concatenate((X, X_temp), axis=1)

        y = np.array(y)
        return X, y


class RandomIndex:
    """
    """
    def __init__(self, percent_train, percent_dev, percent_test, **kwargs):
        """
            rate_train, rate_dev, rate_test should in percentage

            options:
                loglevel
                zero_vector_idxs_filename
                emotions
        """

        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        self.zero_vector_idxs_filename = None if 'zero_vector_idxs_filename' not in kwargs else kwargs['zero_vector_idxs_filename']
        self.emotions = filename.emotions['LJ40K'] if 'emotions' not in kwargs else kwargs['emotions']

        if percent_train + percent_dev + percent_test != 100:
            raise ValueError("percent_train + percent_dev + percent_test should be 100")

        if percent_train == 0:
            raise ValueError("percent_train should not be zero")

        self.rate_train = float(percent_train)/100
        self.rate_dev = float(percent_dev)/100
        self.rate_test = float(percent_test)/100

    def _shuffle_index_of_sets(self, ndoc):

        ntrain = long(ndoc * self.rate_train)
        ndev = long(ndoc * self.rate_dev)
        ntest = long(ndoc * self.rate_test)

        nremain = ndoc - ntrain - ndev - ntest
        if nremain != 0:
            # we put the remains in the training set
            ntrain += nremain

        all_idx = range(ndoc)
        random.shuffle(all_idx)

        train = all_idx[:ntrain]
        dev = all_idx[ntrain:ntrain+ndev]
        test = all_idx[ntrain+ndev:ntrain+ndev+ntest]

        return (train, dev, test)

    def _get_random_subset(self, entire_set, nsubset):
        dup = entire_set
        random.shuffle(dup)
        return dup[:nsubset]

    def _remove_zero_vector_index(self, train, dev, test, zv_idxs):

        for emotion in self.emotions:
            train[emotion][emotion] = filter(lambda x: x not in zv_idxs[emotion], train[emotion][emotion])
            dev[emotion][emotion] = filter(lambda x: x not in zv_idxs[emotion], dev[emotion][emotion])
            test[emotion][emotion] = filter(lambda x: x not in zv_idxs[emotion], test[emotion][emotion])

    def shuffle(self, root):
        """
            return three lists (train_idx, dev_idx, test_idx)
        """

        # 2D dictionary to store the sets for each emotion
        train = {}
        dev = {}
        test = {}

        # generate positive indices
        for emotion in self.emotions:

            emotion_dir = os.path.join(root, emotion)
            ndoc = len(os.listdir(emotion_dir)) - 2     # minus . and ..
            self.logger.info("emotion = %s, ndoc = %u", emotion, ndoc)
            
            train[emotion] = {}
            dev[emotion] = {}
            test[emotion] = {}
            train[emotion][emotion], dev[emotion][emotion], test[emotion][emotion] = self._shuffle_index_of_sets(ndoc)

            self.logger.debug("len(train[%s][%s]) = %u" % (emotion, emotion, len(train[emotion][emotion])))
            self.logger.debug("len(dev[%s][%s]) = %u" % (emotion, emotion, len(dev[emotion][emotion]))) 
            self.logger.debug("len(test[%s][%s]) = %u" % (emotion, emotion, len(test[emotion][emotion]))) 

        # remove the zero vectors collected by Sven for LJ2M
        if self.zero_vector_idxs_filename != None:
            self._remove_zero_vector_index(train, dev, test, pickle.load(open(self.zero_vector_idxs_filename)))

        """
            shuffle negatives
            algorithm (for all train/dev/test):
                1. we aim to find the same number of negative examples as positives
                2. n_pos / 39 = number of negative examples retrieved from each of other 39 emotions
                3. stack 39 groups together as the negatives
        """
        for emotion in self.emotions:
            
            # the difference between pos. and neg. will be less than 39
            n_neg_train = len(train[emotion][emotion]) / 39    
            n_neg_dev = len(dev[emotion][emotion]) / 39 
            n_neg_test = len(test[emotion][emotion]) / 39 

            for neg_emotion in filter(lambda x: x != emotion, self.emotions):

                train[emotion][neg_emotion] = self._get_random_subset(train[neg_emotion][neg_emotion], n_neg_train)
                dev[emotion][neg_emotion] = self._get_random_subset(dev[neg_emotion][neg_emotion], n_neg_dev)
                test[emotion][neg_emotion] = self._get_random_subset(test[neg_emotion][neg_emotion], n_neg_test)

                self.logger.debug("len(train[%s][%s]) = %u" % (emotion, emotion, len(train[emotion][emotion])))
                self.logger.debug("len(train[%s][%s]) = %u" % (emotion, neg_emotion, len(train[emotion][neg_emotion])))

        return train, dev, test
