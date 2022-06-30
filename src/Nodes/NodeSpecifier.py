from abc import ABC, abstractmethod

class NodeSpecifier(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def define_Ar(self, segmentation_map, labelled_image, regions, matching):
        pass
    
    @abstractmethod
    def evaluation_metrics(self, value1, value2):
        pass