import numpy as np
import math
from numpy.linalg import norm

class Ke_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_Ke(self, specifier, Am, Ar, params):
        return self.constructor(specifier, Am, Ar, params)

class Functor(object):
        
    def __call__(self, specifier, Am, Ar, params):
        xm, ym, zm = Am.shape
        xi, yi, zi = Ar.shape

        nb_columns_rows = ym * yi

        Ke = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            for j in range(0,nb_columns_rows):
                # Case dans ma matrice K
                indice_case_row = math.floor(i / yi)
                indice_case_column = math.floor(j / yi)

                # Position dans la case
                indice_case_inside_row = i % yi
                indice_case_inside_column = j % yi

                # Obtention des vecteur des positions
                value_edge_model = Am[:, indice_case_row, indice_case_column]
                value_edge_test = Ar[:, indice_case_inside_row, indice_case_inside_column]

                if specifier == "centroid":
                    Ke[i][j] = self. __Centroid(value_edge_model, value_edge_test, params)
                elif specifier == "edt_min":
                    Ke[i][j] = self. __EDT_min(value_edge_model, value_edge_test, params)

        return Ke
    
    def __Centroid(self, value_edge_model, value_edge_test, params):
        
        # Evaluation des dissimilarités
        norm1 = norm(value_edge_model)
        norm2 = norm(value_edge_test)

        scalar_pro = np.dot(value_edge_model, value_edge_test)

        if norm1 !=0 and norm2 !=0:
            cos_angle = scalar_pro / ( norm1 * norm2)

            return params["lbd"] * math.fabs(cos_angle - 1)/2 + (1-params["lbd"]) * math.fabs(norm1 - norm2) / params["Cs"]
        else:
            return 0

    def __EDT_min(self, value_edge_model, value_edge_test, params):
        
        if 0 in np.unique(value_edge_model) and 0 in np.unique(value_edge_test) :
            return 0
        else:
            return (params["lbd"] * math.fabs(value_edge_model[0] - value_edge_test[0]) + (1-params["lbd"]) *  math.fabs(value_edge_model[1] - value_edge_test[1])) / params["Cs"] 