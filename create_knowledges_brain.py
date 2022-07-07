import math
import os
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
import random

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


def create_adjency_matrix(node_kowledge_constructor, label_path, parameters):
    
    files = os.listdir(label_path)

    files = random.sample(files, math.floor(len(files) / 2))

    Ae_temp = {}
    Ar_temp = {}

    for f in files:
        file_path = os.path.join(label_path, f)
        image = load_file(file_path)
        
        Ae_temp = add_values_on_specifier(edges_knowledge_constructor, Ae_temp, edges_knowledge_constructor.get_knowledges(image, parameters))
        Ar_temp = add_values_on_specifier(node_kowledge_constructor, Ar_temp, node_kowledge_constructor.get_knowledges(image))

    Ar = average_results_on_specifier(node_kowledge_constructor, Ar_temp)
    Ae = average_results_on_specifier(edges_knowledge_constructor, Ae_temp)
    
    return Ar, Ae

# Parameters

parameters = {
    'weigthed' : True
}

train_labels_path = os.path.join('data', "annotations")

experiment_path = 'experiments'

NB_CLASSES = 3

nodes_specifier = [
    CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier(),
    MaxDistanceSpecifier.MaxDistandeSpecifier()
]

node_knowledge_constructor = KnowledgesNodesConstructor(specifier_list=nodes_specifier)

edges_specifier = [
    RelativePositionSpecifier.RelativePositionSpecifier(),
    MinMaxEdtDistanceSpecifier.MinMaxEdtDistandeSpecifier()
]

edges_knowledge_constructor = KnowledgesEdgesConstructor(specifier_list=edges_specifier)

# Methodes

local_knowlege_path = os.path.join(experiment_path, "knowledges")
if os.path.isdir(local_knowlege_path) is False:
    os.mkdir(local_knowlege_path)
    
local_node_knowlege_path = os.path.join(local_knowlege_path, "nodes")
if os.path.isdir(local_node_knowlege_path) is False:
    os.mkdir(local_node_knowlege_path)
    
local_edge_knowlege_path = os.path.join(local_knowlege_path, "edges")
if os.path.isdir(local_edge_knowlege_path) is False:
    os.mkdir(local_edge_knowlege_path)

Ar, Ae = create_adjency_matrix(node_knowledge_constructor, train_labels_path, parameters)

for specifier in node_knowledge_constructor.specifier_list:
    Ar_specifier_path = os.path.join(local_knowlege_path, "nodes","{}.npy".format(specifier.name))
    np.save(Ar_specifier_path, Ar[specifier.name])

for specifier in edges_knowledge_constructor.specifier_list:
    Ae_specifier_path = os.path.join(local_knowlege_path, "edges","{}.npy".format(specifier.name))
    np.save(Ae_specifier_path, Ae[specifier.name])