from abc import ABC, abstractmethod


class EdgeSpecifier(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def define_Ae(self, segmentation_map, labelled_image, regions, matching, parameters):
        pass
    
    @abstractmethod
    def evaluation_metrics(self, value1, value2, parameters):
        pass