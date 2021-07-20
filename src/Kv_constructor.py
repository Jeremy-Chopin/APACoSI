import numpy as np
import math
from sklearn.metrics import mean_squared_error

class Kv_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_Kv(self, specifier, nodes_model, nodes_unsure, Am, Ar):
        return self.constructor(specifier, nodes_model, nodes_unsure, Am, Ar)

class Functor(object):
        
    def __call__(self, specifier, nodes_model, nodes_unsure, Am, Ar):
        xm, ym = Am[0,:,:].shape
        xi, yi = Ar[0,:,:].shape

        nb_columns_rows = ym * yi

        Kv = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            
            # position dans la matrice Kv
            indice_model = math.floor(i / xi)
            indice_test = i % xi

            # Evaluation des dissimilarités
            value_node_model = nodes_model[indice_model].probability_vector
            value_node_test = nodes_unsure[indice_test].probability_vector

            if specifier == "centroid":
                Kv[i][i] = self. __Centroid(value_node_model, value_node_test)
            elif specifier == "edt_min":
                Kv[i][i] = self. __EDT_min(value_node_model, value_node_test)

        return Kv
    
    def __Centroid(self, value_node_model, value_node_test):
        return mean_squared_error(value_node_model, value_node_test)

    def __EDT_min(self, value_node_model, value_node_test):
        return mean_squared_error(value_node_model, value_node_test)
