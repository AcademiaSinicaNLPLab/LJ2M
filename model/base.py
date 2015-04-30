import abc

class LearnerBase:
    __metaclass__ = abc.ABCMeta

    #def __init__(self):
    #    raise NotImplementedError("This is an abstract class.")

    @abc.abstractmethod
    def set(self, X, y, feature_name):
        """
            set the training X, y, and feature name string
        """
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, X_test, y_test, **kwargs):
        pass

    @abc.abstractmethod
    def dump_model(self, file_name):
        pass

    @abc.abstractmethod
    def load_model(self, file_name):
        pass

