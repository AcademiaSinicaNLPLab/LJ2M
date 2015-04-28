
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
    def __init__(self, idx_dict, data_path, **kwargs):
        """
            idx_dict[emotion]['train']
            idx_dict[emotion]['dev']
            idx_dict[emotion]['test']

            options:
                loglevel
        """
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        self.idx_dict = idx_dict
        self.Xs_pos = {}
        self.Xs_neg = {}

        for emotion in filename.emotions['LJ40K']:
            fpath = FeatureList.get_full_data_path(emotion, data_path)

            self.logger.info("load features from %s", fpath)
            try:
                Xy = pickle.load(open(fpath, "r"))
            except ValueError:
                self.logger.error("failed to load %s" % (fpath))

            X = np.zeros((len(Xy),300),dtype="float32")
            # ToDo: concatenating numpy array wastes too many times, so we need do it when generating pkl
            for i in range(len(Xy)):    
                # make sure only one feature vector in each doc
                assert Xy[i]['X'].shape[0] == 1                
                X[i] = Xy[i]['X']
                self.logger.info("X.shapae = %s", X.shape)

            X_dict['train'] = X[self.idx_dict[emotion]['train']]
            X_dict['dev'] = X[self.idx_dict[emotion]['dev']]
            X_dict['test'] = X[self.idx_dict[emotion]['test']]
            self.Xs_pos[emotion] = X_dict

            import pdb; pdb.set_trace()

    def _get_negative_examples(self, emotion_pos, set_type):
        """
            algorithm (for all train/dev/test):
                1. we aim to find the same number of negative examples as positives
                2. n_pos / 39 = number of negative exampel retrieved from each of other 39 emotions
                3. stack 39 groups together
        """
        pass

    def _get_set_by_emotions(self, emotion, set_type):
        assert len(self.Xs) != 0 and len (self.Xs[emotion]) != 0

        negs = self._get_negative_examples(emotion, set_type)

        #negs
        #+
        #self.Xs_pos[emotion][set_type]

        #return self.Xs_pos[emotion][set_type]

    def get_training_set_by_emotion(self, emotion):
        return self._get_set_by_emotions(emotion, 'train')

    def get_dev_set_by_emotion(self, emotion):
        return self._get_set_by_emotions(emotion, 'dev')

    def get_testing_set_by_emotion(self, emotion):
        return self._get_set_by_emotions(emotion, 'test')

    def get_dataset_by_emotion(self, emotion):
        return (self.get_training_set_by_emotion(emotion), self.get_dev_set_by_emotion(emotion), self.get_testing_set_by_emotion(emotion))

    def _get_exclusive_emotions_set(self, set_type):
        other_emotions = utils.get_unique_list_diff(filename.emotions['LJ2M'], filename.emotions['LJ40K'])

        #for oe in other_emotions:
        #    data = _get_set_by_emotions(oe, set_type)

    def get_exclusive_emotions_train(self):
        return self._get_exclusive_emotions_set('train')

    def get_exclusive_emotions_dev(self):
        return self._get_exclusive_emotions_set('dev')

    def get_exclusive_emotions_test(self):
        return self._get_exclusive_emotions_set('test')


class FusedDataset:
    """
        fuse different features
    """
    def __init__(self, **kwargs):
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

        self.feature_name = []
        self.datasets = []

    def add_feature(self, feature_name, dataset):
        self.feature_name.append(feature_name)
        self.datasets.append(dataset)

    def get_training_set_by_emotion(self, emotion):
        pass

    def get_dev_set_by_emotion(self, emotion):
        pass

    def get_testing_set_by_emotion(self, emotion):
        pass

    def get_dataset_by_emotion(self, emotion):
        return (self.get_training_set_by_emotion(emotion), self.get_dev_set_by_emotion(emotion), self.get_testing_set_by_emotion(emotion))


class Fuser:    # ToDo: merge Dataset instances
    """
    Fuse features from .npz files
    usage:
        >> from feelit.features import DataPreprocessor
        >> import json
        >> features = ['TFIDF', 'keyword', 'xxx', ...]
        >> dp = DataPreprocessor()
        >> dp.loads(features, files)
        >> X, y = dp.fuse()
    """
    def __init__(self, **kwargs):
        """
        options:
            logger: logging instance
        """
        self.clear()

        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

    def loads(self, features, paths):
        """
        Input:
            paths       : list of files to be concatenated
            features:   : list of feature names
        """
        for i, path in enumerate(paths):
            self.logger.info('loading data from %s' % (path))
            data = np.load(path)            

            #X =  self.replace_nan( self.full_matrix(data['X']) )
            X = data['X']
            self.Xs[features[i]] = X
            self.ys[features[i]] = data['y'];

            self.logger.info('feature "%s", %dx%d' % (features[i], X.shape[0], X.shape[1]))

            self.feature_name.append(features[i])

    def fuse(self):
        """
        Output:
            fused (X, y) from (self.Xs, self.ys)
        """

        # try two libraries for fusion
        try:
            X = np.concatenate(self.Xs.values(), axis=1)
        except ValueError:
            from scipy.sparse import hstack
            candidate = tuple([arr.all() for arr in self.Xs.values()])
            X = hstack(candidate)
              
        y = self.ys[ self.ys.keys()[0] ]

        # check all ys are same  
        for k, v in self.ys.items():
            assert (y == v).all()
        feature_name = '+'.join(self.feature_name)

        self.logger.debug('fused feature name is "%s", %dx%d' % (feature_name, X.shape[0], X.shape[1]))

        return X, y, feature_name

    def clear(self):
        self.Xs = {}
        self.ys = {}
        self.feature_name = []

    def get_binary_y_by_emotion(self, y, emotion):
        '''
        return y with elements in {1,-1}
        '''       
        yb = np.array([1 if val == emotion else -1 for val in y])
        return yb

    def get_examples_by_polarities(self, X, y):
        """
            input:  X: feature vectors
                    y: should be a list of 1 or -1
            output: (positive X, negative X)
        """
        idx_pos = [i for i, v in enumerate(y) if v==1]
        idx_neg = [i for i, v in enumerate(y) if v<=0]
        return X[idx_pos], X[idx_neg]




class RandomIndex:
    """
    """
    def __init__(self, percent_train, percent_dev, percent_test, **kwargs):
        """
            rate_train, rate_dev, rate_test should in percentage
        """

        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__)

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

    def shuffle(self, root, emotion_dirs):
        """
            return three lists (train_idx, dev_idx, test_idx)
        """

        # 2D dictionary to store the sets for each emotion
        train = {}
        dev = {}
        test = {}

        # generate positive indices
        for emotion in emotion_dirs:

            emotion_dir = os.path.join(root, emotion)
            ndoc = len(os.listdir(emotion_dir)) - 2     # minus . and ..
            self.logger.info("emotion = %s, ndoc = %u", emotion, ndoc)
            
            train[emotion] = {}
            dev[emotion] = {}
            test[emotion] = {}
            train[emotion][emotion], dev[emotion][emotion], test[emotion][emotion] = self._shuffle_index_of_sets(ndoc)

        """
            shuffle negatives
            algorithm (for all train/dev/test):
                1. we aim to find the same number of negative examples as positives
                2. n_pos / 39 = number of negative examples retrieved from each of other 39 emotions
                3. stack 39 groups together as the negatives
        """
        for emotion in emotion_dirs:
            
            # the difference between pos. and neg. will be less than 39
            n_neg_train = len(train[emotion][emotion]) / 39    
            n_neg_dev = len(dev[emotion][emotion]) / 39 
            n_neg_test = len(test[emotion][emotion]) / 39 

            for neg_emotion in emotion_dirs:

                if neg_emotion == emotion:
                    continue

                train[emotion][neg_emotion] = self._get_random_subset(train[neg_emotion][neg_emotion], n_neg_train)
                dev[emotion][neg_emotion] = self._get_random_subset(dev[neg_emotion][neg_emotion], n_neg_dev)
                test[emotion][neg_emotion] = self._get_random_subset(test[neg_emotion][neg_emotion], n_neg_test)

                self.logger.debug("len(train[%s][%s]) = %u" % (emotion, emotion, len(train[emotion][emotion])))
                self.logger.debug("len(train[%s][%s]) = %u" % (emotion, neg_emotion, len(train[emotion][neg_emotion])))

        return train, dev, test
