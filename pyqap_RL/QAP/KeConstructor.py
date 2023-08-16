import numpy as np
import math
import time

class KeConstructor():
    
    def __init__(self, specifier_list, specifier_weigths, knowledges):
        self.specifier_list = specifier_list
        self.specifier_weigths = specifier_weigths
        self.Aes = {}
        self.knowledges = knowledges
    
    # initial matching

    def construct_Ke_initial(self, segmentation_map, labelled_image, regions, parameters, nb_classes):
        
        self.__define_nodes_informations_initial(segmentation_map, labelled_image, regions, parameters, nb_classes)
        
        Ke = None
        
        for i in range(0, len(self.specifier_list)):
            
            K = self.__construct_specifier_matrice_initial(self.specifier_list[i], segmentation_map, labelled_image, regions, parameters, nb_classes)
            
            if Ke is None:
                Ke = self.specifier_weigths[i] * K
            else:
                Ke += self.specifier_weigths[i] * K
                
        return Ke
    
    def __define_nodes_informations_initial(self, segmentation_map, labelled_image, regions, parameters, nb_classes):
        
        for specifier in self.specifier_list:
            
            Ae = specifier.define_Ae_initial(segmentation_map, labelled_image, regions, parameters, nb_classes)
            self.Aes[specifier.name] = Ae
            
    def __construct_specifier_matrice_initial(self, specifier, segmentation_map, labelled_image, regions, parameters, nb_classes):
        nb_regions = len(regions)

        nb_columns_rows = nb_regions * nb_classes

        K = np.zeros((nb_columns_rows, nb_columns_rows))

        for i in  range(0, nb_columns_rows):
            for j in  range(0, nb_columns_rows):
                if i!=j:
                
                    # Case dans ma matrice K
                    indice_case_row = math.floor(i / nb_regions)
                    indice_case_column = math.floor(j / nb_regions)

                    # Position dans la case
                    indice_case_inside_row = i % nb_regions
                    indice_case_inside_column = j % nb_regions         
                    
                    value_model = self.knowledges[specifier.name][indice_case_row][indice_case_column]
                    value_image = self.Aes[specifier.name][indice_case_inside_row][indice_case_inside_column]
                    
                    K[i][j] = specifier.evaluation_metrics(value_model, value_image, parameters)
        
            
        return K
        
    # refinement
            
    def construct_Ke(self, segmentation_map, labelled_image, regions, matching, parameters, label_to_update = None, ke = None):
        
        self.__define_nodes_informations(segmentation_map, labelled_image, regions, matching, parameters, label_to_update)
        
        Ke = None
        
        for i in range(0, len(self.specifier_list)):
            
            if label_to_update is None:
                K = self.__construct_specifier_matrice(self.specifier_list[i], segmentation_map, labelled_image, regions, matching, parameters)
            else:
                K = self.__construct_specifier_matrice_refinement(self.specifier_list[i], segmentation_map, labelled_image, regions, matching, parameters, label_to_update, ke)

            if Ke is None:
                Ke = self.specifier_weigths[i] * K
            else:
                Ke += self.specifier_weigths[i] * K
                
        return Ke
    
    def __define_nodes_informations(self, segmentation_map, labelled_image, regions, matching, parameters, label_to_update):
        
        for specifier in self.specifier_list:
            if label_to_update is None:
                Ae = specifier.define_Ae(segmentation_map, labelled_image, regions, matching, parameters)
            else:
                Ae = specifier.define_Ae_refinement(segmentation_map, labelled_image, regions, matching, parameters, label_to_update, self.Aes[specifier.name])
            
            self.Aes[specifier.name] = Ae
            
    def __construct_specifier_matrice(self, specifier, segmentation_map, labelled_image, regions, matching, parameters):
        nb_classes = len(matching)

        nb_columns_rows = nb_classes * nb_classes

        K = np.zeros((nb_columns_rows, nb_columns_rows))

        t0 = time.perf_counter()
        for i in  range(0, nb_columns_rows):
            for j in  range(0, nb_columns_rows):
                if i!=j:
                
                    # Position dans la case
                    indice_case_inside_row = i % nb_classes
                    indice_case_inside_column = j % nb_classes  

                    # Case dans ma matrice K
                    indice_case_row = math.floor(i / nb_classes)
                    indice_case_column = math.floor(j / nb_classes)    
                    
                    value_model = self.knowledges[specifier.name][indice_case_row][indice_case_column]
                    value_image = self.Aes[specifier.name][indice_case_inside_row][indice_case_inside_column]
                    
                    K[i][j] = specifier.evaluation_metrics(value_model, value_image, parameters)
        print("Ke refinement: ", time.perf_counter() - t0)
            
        return K

    def __construct_specifier_matrice_refinement(self, specifier, segmentation_map, labelled_image, regions, matching, parameters, label_to_reduce, K):
        nb_classes = len(matching)

        nb_columns_rows = nb_classes * nb_classes

        for i in  range(0, nb_columns_rows):
            for j in  range(0, nb_columns_rows):
                if i!=j:
                
                    # Position dans la case
                    indice_case_inside_row = i % nb_classes
                    indice_case_inside_column = j % nb_classes

                    if indice_case_inside_column == label_to_reduce or indice_case_inside_row == label_to_reduce:

                        # Case dans ma matrice K
                        indice_case_row = math.floor(i / nb_classes)
                        indice_case_column = math.floor(j / nb_classes) 
                        
                        value_model = self.knowledges[specifier.name][indice_case_row][indice_case_column]
                        value_image = self.Aes[specifier.name][indice_case_inside_row][indice_case_inside_column]
                        
                        K[i][j] = specifier.evaluation_metrics(value_model, value_image, parameters)
            
        return K