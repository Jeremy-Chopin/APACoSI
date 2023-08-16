import numpy as np
import pandas as pd
import os

from ..utils import load_file

def create_adjency_matrix(node_kowledge_constructor, parameters, files_path):
    
    Ar_temp = {}

    for file_path in files_path:
        image = load_file(file_path)
        
        Ar_temp = __add_values_on_specifier(node_kowledge_constructor, Ar_temp, node_kowledge_constructor.get_knowledges(image, parameters))

    Ar = __average_results_on_specifier(node_kowledge_constructor, Ar_temp)
    
    return Ar

def __add_values_on_specifier(node_knowledge_constructor, actual_dict, dict_to_add):
    
    for specifier in node_knowledge_constructor.specifier_list:
        
        if specifier.name not in actual_dict.keys():
            actual_dict[specifier.name] = []
            
        actual_dict[specifier.name].append(dict_to_add[specifier.name])
        
    return actual_dict

def __average_results_on_specifier(node_knowledge_constructor, actual_dict):
    
    for specifier in node_knowledge_constructor.specifier_list:
        
        mean_array = np.stack(actual_dict[specifier.name], axis=0)
        mean_array = np.mean(mean_array, axis=0)
        
        actual_dict[specifier.name] = mean_array
        
    return actual_dict