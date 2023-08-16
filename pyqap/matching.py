import numpy as np
import math
from copy import deepcopy

from skimage.measure import label, regionprops

from pyqap.KvConstructor import KvConstructor
from pyqap.KeConstructor import KeConstructor



def define_permutations(regions, nb_classes):
    """Algorithms to define the permutations that will be used with the QAP on a one-to-one matching.

    Args:
        regions (list): the regions obtain from scipy regionprops applied to the image.
        nb_classes (int): the number of classes of the system.

    Returns:
        list: A list of numpy array describing all the permutations possibilities.
    """

    all_permutations = []
    
    initial_regions_labels = np.zeros((len(regions), nb_classes))
        
    indice = 0
    for region in regions:
        label = region.max_intensity
        max_id = label - 1
        initial_regions_labels[indice][int(max_id)] = 1
        indice += 1

    dico = {}

    for i in range(0, nb_classes):
        if np.sum(initial_regions_labels[:,i]) > 1:
            indices = np.squeeze(np.where(initial_regions_labels[:,i] != 0), axis=0)
            temp = []
            for indice in indices:
                t =  np.zeros(len(regions))
                t[indice] = 1
                temp.append(t)
            dico[i] = temp

    all_permutations.append(initial_regions_labels)

    for k, v in dico.items():

        temp = []
        while(len(all_permutations) != 0):
            val = all_permutations.pop()

            for row in v:
                val[:,k] = row
                temp.append(np.copy(val))
        all_permutations = temp
        
    return all_permutations

def permutation_to_matching(permutation):
    
    nb_regions, nb_classes = permutation.shape
    
    matching = {}
    
    for i in range(0, nb_classes):
        
        matching[i] = list(np.where(permutation[:,i] == 1)[0])
        
    return matching





def evaluate_matching(matching, kv, ke, alpha, nb_classes):
    assert ke is not None or kv is not None
            
    if kv is None:
        K = ke
    elif ke is None:
        K = kv
    else:
        K = alpha * kv + (1-alpha) * ke
    
    X = np.zeros((nb_classes, nb_classes))
    
    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i == j:
                X[i][j] = 1

    vec_x = X.flatten('F')

    x_translate = np.transpose(vec_x)
    tempo = np.matmul(x_translate,K)
    score = np.dot(tempo,vec_x)
    
    return score

def get_one_to_one_matching(nb_classes, params, image_cnn, pr_mask, node_knowledge, edge_knowledge, nodes_specifier, edges_specifier, nodes_specifier_weigths, edges_specifier_weigths):
    labelled_image = label(image_cnn)
    regions = regionprops(labelled_image, image_cnn)
    
    M = define_permutations(regions, nb_classes)
    
    best_matching = None
    best_score = math.inf
    for permutation in M:
        
        matching = permutation_to_matching(permutation)
    
        kv_constructor = KvConstructor(nodes_specifier, nodes_specifier_weigths, node_knowledge)
        
        kv = kv_constructor.construct_Kv(pr_mask, labelled_image, regions, matching, params)
        
        ke_constructor = KeConstructor(edges_specifier, edges_specifier_weigths, edge_knowledge)
        
        ke = ke_constructor.construct_Ke(pr_mask, labelled_image, regions, matching, params)
        
        score = evaluate_matching(matching, kv, ke, params['alpha'], nb_classes)
        
        if score < best_score:
            best_score = score
            best_matching = matching

    return best_matching, best_score, labelled_image, regions

def get_many_to_one_matching(nb_classes, params, pr_mask, labelled_image, regions, best_matching, best_score, node_knowledge, edge_knowledge, nodes_specifier, edges_specifier, nodes_specifier_weigths, edges_specifier_weigths):
    list_ids = list(np.unique(labelled_image)[1:])
    
    for k,v in best_matching.items():
        for node in v:
            list_ids.remove(node)
    
    temp_best_matching = deepcopy(best_matching)
    temp_best_score = best_score
    
    for ids in list_ids:
        
        class_best_matching = temp_best_matching
        class_best_score = math.inf
        
        for cls in range(0, nb_classes):
            temp_matching = deepcopy(temp_best_matching)
            
            temp_matching[cls].append(ids)
            
            kv_constructor = KvConstructor(nodes_specifier, nodes_specifier_weigths, node_knowledge)
        
            kv = kv_constructor.construct_Kv(pr_mask, labelled_image, regions, temp_matching, params)
            
            ke_constructor = KeConstructor(edges_specifier, edges_specifier_weigths, edge_knowledge)
            
            ke = ke_constructor.construct_Ke(pr_mask, labelled_image, regions, temp_matching, params)
            
            score = evaluate_matching(temp_matching, kv, ke, params['alpha'], nb_classes)
            
            if score < class_best_score:
                class_best_score = score
                class_best_matching = temp_matching
        
        if class_best_score < temp_best_score:
            temp_best_score = class_best_score
            temp_best_matching = class_best_matching
    
    return temp_best_matching, temp_best_score

