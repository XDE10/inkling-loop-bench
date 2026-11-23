from abc import ABC, abstractmethod

class Backend(ABC):
    @abstractmethod
    def generate(self, question, config):
        raise NotImplementedError
