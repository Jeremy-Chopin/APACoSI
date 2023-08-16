import cv2
import nibabel as nib
import numpy as np
import os
import math

def load_file(path):

    if ".png" in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif ".npy" in path:
        image = np.load(path)
    elif ".nii.gz" in path or ".nii" in path:
        image = nib.load(path)
        image = image.get_fdata()
        image = np.squeeze(image).astype(np.uint8)

    else:
        print("Error !")

    return image


def load_knowledge(knowledge_path, mode, specifier_list):
    
    assert mode == 'nodes' or mode == 'edges'
    
    A = {}
    
    knowledge_path = os.path.join(knowledge_path, mode)
    
    for specifier in specifier_list:
    
        file_path = os.path.join(knowledge_path, '{}.npy'.format(specifier.name))
        A[specifier.name] = np.load(file_path)
        
    return A


def calculate_max_diagonal(image):
	
	axes = list(image.shape)

	somme = 0
	for axe in axes:
		somme += math.pow(axe, 2)
	
	return math.sqrt(somme)