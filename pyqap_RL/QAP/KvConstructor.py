from lib2to3.pytree import Node
import numpy as np
import math
import time

class KvConstructor():
    
    def __init__(self, specifier_list, specifier_weigths, knowledges):
        self.specifier_list = specifier_list
        self.specifier_weigths = specifier_weigths
        self.Ans = {}
        self.knowledges = knowledges

    # initial_matching

    def construct_Kv_initial(self, segmentation_map, labelled_image, regions, parameters, nb_classes):
        
        self.__define_nodes_informations_initial(segmentation_map, labelled_image, regions, parameters, nb_classes)
        
        Kv = None
        
        for i in range(0, len(self.specifier_list)):
            
            K = self.__construct_specifier_matrice_initial(self.specifier_list[i], segmentation_map, labelled_image, regions, parameters, nb_classes)
            
            if Kv is None:
                Kv = self.specifier_weigths[i] * K
            else:
                Kv += self.specifier_weigths[i] * K
                
        return Kv
    
    def __define_nodes_informations_initial(self, segmentation_map, labelled_image, regions, parameters, nb_classes):
        
        for specifier in self.specifier_list:
            An = specifier.define_Ar_initial(segmentation_map, labelled_image, regions, parameters, nb_classes)
            self.Ans[specifier.name] = An
    
    def __construct_specifier_matrice_initial(self, specifier, segmentation_map, labelled_image, regions, parameters, nb_classes):
        nb_regions = len(regions)

        nb_columns_rows = nb_regions * nb_classes

        K = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            
            indice_model = math.floor(i / nb_regions)
            indice_test = i % nb_regions 
            
            value_model = self.knowledges[specifier.name][indice_model][indice_model]
            value_image = self.Ans[specifier.name][indice_test][indice_test]
            
            K[i][i] = specifier.evaluation_metrics(value_model, value_image, parameters)
            
        return K

    # refinement
        
    def construct_Kv(self, segmentation_map, labelled_image, regions, matching, parameters, label_to_update = None):
        
        self.__define_nodes_informations(segmentation_map, labelled_image, regions, matching, parameters, label_to_update)
        
        Kv = None
        
        for i in range(0, len(self.specifier_list)):
            
            K = self.__construct_specifier_matrice(self.specifier_list[i], segmentation_map, labelled_image, regions, matching, parameters)
            
            if Kv is None:
                Kv = self.specifier_weigths[i] * K
            else:
                Kv += self.specifier_weigths[i] * K
                
        return Kv
    
    def __define_nodes_informations(self, segmentation_map, labelled_image, regions, matching, parameters, label_to_update):
        
        for specifier in self.specifier_list:
            if label_to_update is None:
                An = specifier.define_Ar(segmentation_map, labelled_image, regions, matching, parameters)
            else:
                An = specifier.define_Ar_refinement(segmentation_map, labelled_image, regions, matching, parameters, label_to_update, self.Ans[specifier.name])
            self.Ans[specifier.name] = An
    
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