import os
from turtle import distance
import numpy as np
import nibabel as nib
import math

from PIL import Image

def load_knowledge(knowledge_path, mode, specifier_list):
    
    assert mode == 'nodes' or mode == 'edges'
    
    A = {}
    
    knowledge_path = os.path.join(knowledge_path, mode)
    
    for specifier in specifier_list:
    
        file_path = os.path.join(knowledge_path, '{}.npy'.format(specifier.name))
        A[specifier.name] = np.load(file_path)
        
    return A

def load_file(path):
    affine = None
    if ".nii.gz" in path:
        image = nib.load(path)
        affine = image.affine
        image = image.get_fdata()
        image = np.squeeze(image)
    elif '.png' in path:
        image = Image.open(path)
        image = np.array(image)
    elif '.npy' in path:
        image = np.load(path)
    else:
        print("Error !")
    
    return image, affine

def sorting(l1, l2):
    # l1 and l2 has to be numpy arrays
    idx = np.argsort(l1)
    return l1[idx], l2[idx]

def filter_permutation(all_permutations, regions, nb_nodes_max, nb_classes):

    for i in range(0, nb_classes):
        if np.sum(all_permutations[:,i]) > nb_nodes_max:

            x = np.where(all_permutations[:,i] == 1)[0]

            sizes = {}

            for k in x:
                sizes[k] = regions[k].area

            areas, idx = sorting(-np.array(list(sizes.values())), np.array(list(sizes.keys())))



            for j in range(nb_nodes_max,len(idx)):
                all_permutations[idx[j]][i] = 0
    
    return all_permutations





def define_permutations(regions, nb_classes, nb_nodes_max):
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
        initial_regions_labels[indice][max_id] = 1
        indice += 1
    
    initial_regions_labels = filter_permutation(initial_regions_labels, regions, nb_nodes_max, nb_classes)

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

def get_diagonal_size(image):
    
    somme = 0
    for i in range(0, len(image.shape)):
        somme += math.pow(image.shape[i], 2)
        
    distance = math.sqrt(somme)
    
    return distance

def create_images_from_ids(labelled_image, matching):
    
    image_inter = np.zeros(labelled_image.shape)

    for k,v in matching.items():
        for ids in v:
            image_inter = np.where(labelled_image == ids + 1, k + 1, image_inter)
    
    return image_inter