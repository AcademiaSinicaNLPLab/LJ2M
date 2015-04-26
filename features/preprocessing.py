
import logging

def get_feature_list(feature_list_file):
    fp = open(feature_list_file, 'r')
    feature_list = json.load(fp)
    fp.close()
    return feature_list

class Fuser:
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
    def __init__(self, percent_train, percent_dev, percent_test):
        """
            rate_train, rate_dev, rate_test should in percentage
        """
        if percent_train + percent_dev + percent_test != 100:
            raise ValueError("percent_train + percent_dev + percent_test should be 100")

        self.rate_train = float(percent_train)/100
        self.rate_dev = float(percent_dev)/100
        self.rate_test = float(percent_test)/100

    def shuffle(self, ndata):
        """
            return three lists (train_idx, dev_idx, test_idx)
        """
        import random

        ntrain = long(ndata * self.rate_train)
        ndev = long(ndata * self.rate_dev)
        ntest = long(ndata * self.rate_test)

        nremain = ndata - ntrain - ndev - ntest
        if nremain != 0:
            # we put the remains in the training set
            ntrain += nremain

        all_idx = range(ndata)
        random.shuffle(all_idx)

        train = all_idx[:ntrain]
        dev = all_idx[ntrain:ntrain+ndev]
        test = all_idx[ntrain+ndev:ntrain+ndev+ntest]

        return (train, dev, test)


