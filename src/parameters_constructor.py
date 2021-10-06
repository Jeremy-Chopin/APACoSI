import numpy as np
from src import utils

class Parameters_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_Parameters(self, specifier, labelled_image, regions, nodes):
        return self.constructor(specifier, labelled_image, regions, nodes)

class Functor(object):
    def __call__(self, specifier, labelled_image, regions, nodes) :

        if specifier == "centroid":
            return self. __Centroid(labelled_image, regions, nodes)
        elif specifier == "edt_signed":
            return self. __EDT_min(labelled_image, regions, nodes)
        else :
            return self.__Error()

    def __Centroid(self, labelled_image, regions, nodes):
        Cs = utils.calculate_max_diagonal(labelled_image)
        lbd = 0.5

        parameters = {}
        parameters["Cs"] = Cs
        parameters["lbd"] = lbd

        return parameters

    def __EDT_min(self, labelled_image, regions, nodes):
        Cs = utils.calculate_max_diagonal(labelled_image)
        lbd = 0.75

        parameters = {}
        parameters["Cs"] = Cs
        parameters["lbd"] = lbd

        return parameters
    def __Error(self):
        print ("Data is not correct")