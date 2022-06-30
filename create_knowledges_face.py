import math
import os
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
import random
import time

from src.KnowledgesNodesConstructor import KnowledgesNodesConstructor
from src.Nodes import MaxDistanceSpecifier, CnnProbabilitiesSpecifier

from src.KnowledgeEdgesConstructor import KnowledgesEdgesConstructor
from src.Edges import MinMaxEdtDistanceSpecifier, RelativePositionSpecifier



def load_file(path):
    if ".png" in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif ".nii.gz" in path:
        image = nib.load(path)
        image = image.get_fdata()
        image = np.squeeze(image)
    else:
        print("Error !")

    return image

def add_values_on_specifier(node_knowledge_constructor, actual_dict, dict_to_add):
    
    for specifier in node_knowledge_constructor.specifier_list:
        
        if specifier.name not in actual_dict.keys():
            actual_dict[specifier.name] = []
            
        actual_dict[specifier.name].append(dict_to_add[specifier.name])
        
    return actual_dict

def average_results_on_specifier(node_knowledge_constructor, actual_dict):
    
    for specifier in node_knowledge_constructor.specifier_list:
        
        mean_array = np.stack(actual_dict[specifier.name], axis=0)
        mean_array = np.mean(mean_array, axis=0)
        
        actual_dict[specifier.name] = mean_array
        
    return actual_dict


def create_adjency_matrix(node_kowledge_constructor, label_path, parameters, splits_path):
    
    files = pd.read_csv(splits_path, sep=';').values

    Ae_temp = {}
    Ar_temp = {}

    for f in files:
        try:
            file_path = os.path.join(label_path, f[1])
            image = load_file(file_path)
            
            Ae_temp = add_values_on_specifier(edges_knowledge_constructor, Ae_temp, edges_knowledge_constructor.get_knowledges(image, parameters))
            Ar_temp = add_values_on_specifier(node_kowledge_constructor, Ar_temp, node_kowledge_constructor.get_knowledges(image))
        except:
            print("Error on image ! ")

    Ar = average_results_on_specifier(node_kowledge_constructor, Ar_temp)
    Ae = average_results_on_specifier(edges_knowledge_constructor, Ae_temp)
    
    return Ar, Ae

# Parameters

percentages = ['100', '75', '50']
#iterations = ['0', '1', '2']
iterations = ['3']


#percentages = ['75']
#iterations = ['0']

parameters = {
    'weigthed' : True
}

train_labels_path = os.path.join('data', "annotations", 'train')

experiment_path = 'experiments'

NB_CLASSES = 8

for percentage in percentages:

    percentage_path = os.path.join(experiment_path, percentage)

    for iteration in iterations:

        iteration_path = os.path.join(percentage_path, iteration)

        splits_path = os.path.join(iteration_path, 'train_splits.csv')

        nodes_specifier = [
            CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier(),
            #MaxDistanceSpecifier.MaxDistandeSpecifier()
        ]

        node_knowledge_constructor = KnowledgesNodesConstructor(specifier_list=nodes_specifier)

        edges_specifier = [
            #RelativePositionSpecifier.RelativePositionSpecifier(),
            MinMaxEdtDistanceSpecifier.MinMaxEdtDistandeSpecifier()
        ]

        edges_knowledge_constructor = KnowledgesEdgesConstructor(specifier_list=edges_specifier)

        # Methodes

        local_knowlege_path = os.path.join(iteration_path, "knowledges")
        if os.path.isdir(local_knowlege_path) is False:
            os.mkdir(local_knowlege_path)
            
        local_node_knowlege_path = os.path.join(local_knowlege_path, "nodes")
        if os.path.isdir(local_node_knowlege_path) is False:
            os.mkdir(local_node_knowlege_path)
            
        local_edge_knowlege_path = os.path.join(local_knowlege_path, "edges")
        if os.path.isdir(local_edge_knowlege_path) is False:
            os.mkdir(local_edge_knowlege_path)


        t0 = time.perf_counter()

        Ar, Ae = create_adjency_matrix(node_knowledge_constructor, train_labels_path, parameters, splits_path)

        t1 = time.perf_counter() - t0

        print("Time for knowledge : ", t1)

        for specifier in node_knowledge_constructor.specifier_list:
            Ar_specifier_path = os.path.join(local_knowlege_path, "nodes","{}.npy".format(specifier.name))
            np.save(Ar_specifier_path, Ar[specifier.name])

        for specifier in edges_knowledge_constructor.specifier_list:
            Ae_specifier_path = os.path.join(local_knowlege_path, "edges","{}.npy".format(specifier.name))
            np.save(Ae_specifier_path, Ae[specifier.name])