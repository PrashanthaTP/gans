import time
from abc import abstractmethod
from collections import defaultdict
class BaseEngine:
    def __init__(self):
        pass 
    @abstractmethod
    def run(self,*args,**kwargs):
        raise NotImplementedError('run method must be implemented in all classes inheriting from BaseEngine')


