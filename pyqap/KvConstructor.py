from tkinter.messagebox import NO
import numpy as np
import math

class KvConstructor():
    
    def __init__(self, specifier_list, specifier_weigths, knowledges, ):
        self.specifier_list = specifier_list
        self.specifier_weigths = specifier_weigths
        self.Ans = {}
        self.knowledges = knowledges
        
        
    def __define_nodes_informations(self, segmentation_map, labelled_image, regions, matching, parameters):
        
        for specifier in self.specifier_list:
            
            An = specifier.define_Ar(segmentation_map, labelled_image, regions, matching, parameters)
            self.Ans[specifier.name] = An
            
    def construct_Kv(self, segmentation_map, labelled_image, regions, matching, parameters):
        
        self.__define_nodes_informations(segmentation_map, labelled_image, regions, matching, parameters)
        
        Kv = None
        
        for i in range(0, len(self.specifier_list)):
            
            K = self.__construct_specifier_matrice(self.specifier_list[i], segmentation_map, labelled_image, regions, matching, parameters)
            
            if Kv is None:
                Kv = self.specifier_weigths[i] * K
            else:
                Kv += self.specifier_weigths[i] * K
                
        return Kv
            
            
    
    def __construct_specifier_matrice(self, specifier, segmentation_map, labelled_image, regions, matching, parameters):
        nb_classes = len(matching)

        nb_columns_rows = nb_classes * nb_classes

        K = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            
            # position dans la matrice Kv
            indice_model = math.floor(i / nb_classes)
            indice_test = i % nb_classes
            
            value_model = self.knowledges[specifier.name][indice_model][indice_model]
            value_image = self.Ans[specifier.name][indice_test][indice_test]
            
            K[i][i] = specifier.evaluation_metrics(value_model, value_image, parameters)
            
        return K