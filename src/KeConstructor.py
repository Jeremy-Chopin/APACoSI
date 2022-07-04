import numpy as np
import math

class KeConstructor():
    
    def __init__(self, specifier_list, specifier_weigths, knowledges, ):
        self.specifier_list = specifier_list
        self.specifier_weigths = specifier_weigths
        self.Aes = {}
        self.knowledges = knowledges
        
        
    def __define_nodes_informations(self, segmentation_map, labelled_image, regions, matching, parameters):
        
        for specifier in self.specifier_list:
            
            Ae = specifier.define_Ae(segmentation_map, labelled_image, regions, matching, parameters)
            self.Aes[specifier.name] = Ae
            
    def construct_Ke(self, segmentation_map, labelled_image, regions, matching, parameters):
        
        self.__define_nodes_informations(segmentation_map, labelled_image, regions, matching, parameters)
        
        Ke = None
        
        for i in range(0, len(self.specifier_list)):
            
            K = self.__construct_specifier_matrice(self.specifier_list[i], segmentation_map, labelled_image, regions, matching, parameters)
            
            if Ke is None:
                Ke = self.specifier_weigths[i] * K
            else:
                Ke += self.specifier_weigths[i] * K
                
        return Ke
            
            
    
    def __construct_specifier_matrice(self, specifier, segmentation_map, labelled_image, regions, matching, parameters):
        nb_classes = len(matching)

        nb_columns_rows = nb_classes * nb_classes

        K = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            for j in  range(0, nb_columns_rows):
                if i!=j:
                
                    # Case dans ma matrice K
                    indice_case_row = math.floor(i / nb_classes)
                    indice_case_column = math.floor(j / nb_classes)

                    # Position dans la case
                    indice_case_inside_row = i % nb_classes
                    indice_case_inside_column = j % nb_classes              
                    
                    value_model = self.knowledges[specifier.name][indice_case_row][indice_case_column]
                    value_image = self.Aes[specifier.name][indice_case_inside_row][indice_case_inside_column]
                    
                    K[i][j] = specifier.evaluation_metrics(value_model, value_image, parameters)
            
        return K