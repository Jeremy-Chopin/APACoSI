import os
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