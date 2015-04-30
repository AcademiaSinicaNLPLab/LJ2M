import abc

class FeatureBase:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fetch(self, server, collection):
        pass

    @abc.abstractmethod
    def push(self, server, collection):
        pass

    @abc.abstractmethod
    def calculate(self, filename):
        pass

