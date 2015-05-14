

import logging
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from liblinearutil import *

from common import utils
from .base import LearnerBase


class LinearSVM(LearnerBase):

    def __init__(self, X=None, y=None, feature_name='', **kwargs):
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s')
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 
        self.logger.setLevel(loglevel)

        self.set(X, y, feature_name)

    def set(self, X, y, feature_name):
        if X is not None and y is not None:
            self.X, self.y = self._nparray_to_liblinear(X, y)
            self.problem = problem(self.y, self.X)
        else:
            self.problem = None

        self.feature_name = feature_name

    @staticmethod
    def _is_nparray(arr):
        return (type(arr).__module__ == np.__name__)

    def _nparray_to_liblinear(self, X, y):

        if LinearSVM._is_nparray(X) and LinearSVM._is_nparray(y):
            _X = []
            _y = []

            for x_temp in X:
                _X.append(x_temp.tolist())

            for y_temp in y: 
                _y.append(y_temp)

            return _X, _y
        else:
            raise ValueError('input format is not numpy array')


    def train(self, **kwargs):
        """
        optional:
            kernel: 'linear', ...
            C: float; svm parameters
        """
        self.logger.info("%u samples x %u features in X_train" % (len(self.X), len(self.X[0])))
        self.logger.info("%u samples in y_train" % (len(self.y)))

        ## parameters
        kernel = "linear" if 'kernel' not in kwargs else kwargs["kernel"]
        assert kernel == 'linear'
        C = 1.0 if "C" not in kwargs else kwargs["C"]

        self.logger.info('svm parameters: c=%f, kernel=%s' % (C, kernel))

        param_str = '-c %s' % (C)
        params = parameter(param_str)
        self.logger.debug('prams = %s' % (param_str))
        self.model = train(self.problem, params)

    def dump_model(self, file_name):
        try:
            self.logger.info("dumping %s" % (file_name))
            save_model(file_name, self.model)
        except ValueError:
            self.logger.error("failed to dump %s" % (file_name))
        

    def load_model(self, file_name):
        try:
            self.logger.info("loading %s" % (file_name))
            self.model = load_model(file_name)
        except ValueError:
            self.logger.error("failed to load %s" % (file_name))

    @staticmethod
    def _calculate_sigmoid(p_vals):
        p_vals_temp = [plist[0] for plist in p_vals]
        probas = utils.sigmoid(p_vals_temp)
        return probas

    def predict(self, X_test, y_test, **kwargs):
        """
        optional:
            score
            y_predict
            X_predict_prob
            auc (ToDo)
            decision value
        """
        _X, _y = self._nparray_to_liblinear(X_test, y_test)

        p_labels, p_acc, p_vals = predict(_y, _X, self.model)
        results = {}

        # p_labels: a list of predicted labels

        # p_acc: a tuple including accuracy (for classification), mean
        #    squared error, and squared correlation coefficient (for
        #    regression).

        # p_vals: a list of decision values or probability estimates (if '-b 1' 
        #     is specified). If k is the number of classes, for decision values,
        #     each element includes results of predicting k binary-class
        #     SVMs. If k = 2 and solver is not MCSVM_CS, only one decision value 
        #     is returned. For probabilities, each element contains k values 
        #     indicating the probability that the testing instance is in each class.
        #     Note that the order of classes here is the same as 'model.label'
        #     field in the model structure.

        if 'score' in kwargs and kwargs['score'] == True:
            results.update({'score': p_acc[0]/100})
            self.logger.info('score = %f', results['score'])

        if 'y_predict' in kwargs and kwargs['y_predict'] == True:
            results.update({'y_predict': p_labels})
            self.logger.debug('y_predict = %s', str(p_labels))

        if 'X_predict_prob' in kwargs and kwargs['X_predict_prob'] == True:
            probas = LinearSVM._calculate_sigmoid(p_vals)
            results.update({'X_predict_prob': probas.tolist()})
            self.logger.debug('X_predict_prob = %s', str(probas.tolist()))

        if 'auc' in kwargs and kwargs['auc'] == True:
            fpr, tpr, thresholds = roc_curve(_y, p_vals)    # same result as using sigmoid probability for p_vals
            results.update({'auc': auc(fpr, tpr)})
            self.logger.info('auc = %f', results['auc'])

        if 'decision_value' in kwargs and kwargs['decision_value'] == True:
            results.update({'decision_value': p_vals})
            self.logger.debug('decision_value = %s', str(p_vals))

        return results
