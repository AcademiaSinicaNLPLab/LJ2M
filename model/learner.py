

import logging
import pickle
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# not work in PYTHON 2.7?
# class Classifier:
#     def __init__(self):
#         raise NotImplementedError("This is an abstract class.")

#     @abstractmethod
#     def set(self, X, y, feature_name):
#         """
#             set the training X, y, and feature name string
#         """
#         pass

#     @abstractmethod
#     def train(self, **kwargs):
#         pass

#     @abstractmethod
#     def predict(self, X_test, y_test, **kwargs):
#         pass


#class SVM(Classifier):
class SVM:
    """
    usage:
        >> from models import learners
        >> learner = learners.SVM(loglevel=logging.ERROR) 
        >> learner.set(X_train, y_train, feature_name)
        >>
        >> scores = {}
        >> for C in Cs:
        >>  for gamma in gammas:
        >>      score = learner.kFold(kfolder, classifier='SVM', 
        >>                          kernel='rbf', prob=False, 
        >>                          C=c, scaling=True, gamma=gamma)
        >>      scores.update({(c, gamma): score})
        >>
        >> best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]
        >> learner.train(classifier='SVM', kernel='rbf', prob=True, C=best_C, gamma=best_gamma, 
        >>              scaling=True, random_state=np.random.RandomState(0))
        >> results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)
    """

    def __init__(self, X=None, y=None, feature_name='', **kwargs):
        """
        options:
            loglevel
            do_scaling
            with_mean
            with_std
        """
        loglevel = logging.ERROR if 'loglevel' not in kwargs else kwargs['loglevel']
        logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)
        self.logger = logging.getLogger(__name__+'.'+self.__class__.__name__) 

        self.do_scaling = False if 'scaling' not in kwargs else kwargs['scaling']
        self.with_mean = True if 'with_mean' not in kwargs else kwargs['with_mean']
        self.with_std = True if 'with_std' not in kwargs else kwargs['with_std']

        self.X = X
        self.y = y
        self.feature_name = feature_name
        self.kfold_results = []
        self.Xs = {}
        self.ys = {}


    def set(self, X, y, feature_name):
        self.X = X
        self.y = y
        self.feature_name = feature_name

    def train(self, **kwargs):
        self._train(self.X, self.y, **kwargs)

    def _train(self, X_train, y_train, **kwargs):
        """
        required:
            X_train, y_train

        options:
            prob: True/False. Esimate probability during training
            random_state: seed, RandomState instance or None; for probability estimation
            kernel: 'rbf', ...
            C: float; svm parameters
        """
        self.logger.debug("%u samples x %u features in X_train" % ( X_train.shape[0], X_train.shape[1] ))
        self.logger.debug("%u samples in y_train" % ( y_train.shape[0] ))

        if self.do_scaling:
            self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
            ## apply scaling on X
            self.logger.debug("applying a standard scaling with_mean=%d, with_std=%d" % (self.with_mean, self.with_std))
            X_train = self.scaler.fit_transform(X_train)

        ## determine whether using predict or predict_proba
        self.prob = False if 'prob' not in kwargs else kwargs["prob"]
        random_state = None if 'random_state' not in kwargs else kwargs["random_state"]
        
        ## default rbf kernel
        kernel = "rbf" if 'kernel' not in kwargs else kwargs["kernel"]
        ## C: svm param, default 1
        C = 1.0 if "C" not in kwargs else kwargs["C"]
        ## gamma: rbf param, default (1/num_features)
        num_features = X_train.shape[1]
        gamma = (1.0/num_features) if "gamma" not in kwargs else kwargs["gamma"]
        
        # we use weighted classifier because we might use skewed positive/negative data
        #self.clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=self.prob, random_state=random_state, class_weight='auto')
        self.clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=self.prob, random_state=random_state)
        self.logger.debug("%s C=%f gamma=%f probability=%d" % (kernel, C, gamma, self.prob))

        self.clf.fit(X_train, y_train)
    
    def dump_model(self, file_name):
        try:
            pickle.dump(self.clf, open(file_name, "w"))
        except ValueError:
            self.logger.error("failed to dump %s" % (file_name))

    def dump_scaler(self, file_name):
        try:
            if self.scaling:
                pickle.dump(self.scaler, open(file_name, "w"))
            else:
                self.logger.warning("scaler doesn't exist")
        except ValueError:
            self.logger.error("failed to dump %s" % (file_name))

    def load_model(self, file_name):
        try:
            self.clf = pickle.load(open(file_name, "r"))
        except ValueError:
            self.logger.error("failed to load %s" % (file_name))

    def load_scaler(self, file_name):
        try:
            self.scaler = pickle.load(open(file_name, "r"))
            if self.scaler:
                self.scaling = True
        except ValueError:
            self.logger.error("failed to load %s" % (file_name))

    def predict(self, X_test, y_test, **kwargs):
        '''
        return dictionary of results

        options:
            score
            weighted_score
            y_predict
            X_predict_prob
            auc
        '''
        
        if self.do_scaling:
            self.logger.debug('scaler transforms X_test')
            X_test = self.scaler.transform(X_test)

        self.logger.info('y_test = %s', str(y_test.shape))
        y_predict = self.clf.predict(X_test)
        X_predict_prob = self.clf.predict_proba(X_test) if self.clf.probability else 0
        results = {}
        if 'score' in kwargs and kwargs['score'] == True:
            results.update({'score': self.clf.score(X_test, y_test.tolist())})
            self.logger.info('score = %f', results['score'])

        if 'weighted_score' in kwargs and kwargs['weighted_score'] == True:
            results.update({'weighted_score': self._weighted_score(y_test.tolist(), y_predict)})
            self.logger.info('weighted_score = %f', results['weighted_score'])

        if 'y_predict' in kwargs and kwargs['y_predict'] == True:
            results.update({'y_predict': y_predict})
            self.logger.info('y_predict = %f', results['y_predict'])

        if 'X_predict_prob' in kwargs and kwargs['X_predict_prob'] == True:            
            results.update({'X_predict_prob': X_predict_prob[:, 1]})
            self.logger.info('X_predict_prob = %s', str(results['X_predict_prob']))

        if 'auc' in kwargs and kwargs['auc'] == True:
            fpr, tpr, thresholds = roc_curve(y_test, X_predict_prob[:, 1])
            results.update({'auc': auc(fpr, tpr)})
            self.logger.info('auc = %f', results['auc'])

        return results     
    
    def _weighted_score(self, y_test, y_predict):
        # calc weighted score 
        n_pos = len([val for val in y_test if val == 1])
        n_neg = len([val for val in y_test if val == -1])
        
        temp_min = min(n_pos, n_neg)
        weight_pos = 1.0/(n_pos/temp_min)
        weight_neg = 1.0/(n_neg/temp_min)
        
        correct_predict = [i for i, j in zip(y_test, y_predict) if i == j]
        weighted_sum = 0.0
        for answer in correct_predict:
            weighted_sum += weight_pos if answer == 1 else weight_neg
        
        wscore = weighted_sum / (n_pos * weight_pos + n_neg * weight_neg)
        return wscore
    
    def kfold(self, kfolder, **kwargs):
        """
        return:
            mean score for kfold training

        required:
            kfolder: generated by sklearn.cross_validatio.KFold

        options:
            same as _train 
        """

        sum_score = 0.0
        for (i, (train_index, test_index)) in enumerate(kfolder):

            self.logger.info("cross-validation fold %d: train=%d, test=%d" % (i, len(train_index), len(test_index)))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]

            self._train(X_train, y_train, **kwargs)

            score = self.predict(X_test, y_test, score=True)['score']
            self.logger.info('score = %.5f' % (score))
            sum_score += score

        mean_score = sum_score/len(kfolder)
        self.logger.info('*** C = %f, mean_score = %f' % (kwargs['C'], mean_score))
        return mean_score


