from PIL import Image
from skimage.measure import label, regionprops
from copy import deepcopy
from scipy.ndimage.filters import median_filter
import itk
from matplotlib import pyplot as plt

from torch import tensor, softmax

import time
import os
import math
import numpy as np
import nibabel as nib
import pandas as pd

from src.KvConstructor import KvConstructor
from src.Nodes import MaxDistanceSpecifier, CnnProbabilitiesSpecifier

from src.KeConstructor import KeConstructor
from src.Edges import MinMaxEdtDistanceSpecifier, RelativePositionSpecifier

from src.utils import load_knowledge, load_file, define_permutations, permutation_to_matching, get_diagonal_size, create_images_from_ids
from src import metrics

from progress import bar

def evaluate_matching(matching, kv, ke):
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

def evaluate_permutation(perm, kv, ke):
    assert ke is not None or kv is not None
            
    if kv is None:
        K = ke
    elif ke is None:
        K = kv
    else:
        K = alpha * kv + (1-alpha) * ke
    
    X = perm

    vec_x = X.flatten('F')

    x_translate = np.transpose(vec_x)
    tempo = np.matmul(x_translate,K)
    score = np.dot(tempo,vec_x)
    
    return score

# configuration des chemins

#percentages = ['100', '75', '50']
percentages = ['75']

#iterations = ['0', '1', '2', '3']
iterations = ['0']

#architectures = ['U_Net', 'U_Net_CRF', 'Efficient_U_Net', 'Classic_PSPNet']
architectures = ['Classic_PSPNet']

for architecture in architectures:
    for percentage in percentages:
        for iteration in iterations :
                    
            DATA_PATH = os.path.join('experiments', percentage, iteration, architecture)

            RGB_PATH = os.path.join('data', 'images', 'test')
            LABELS_PATH = os.path.join('data', 'annotations', 'test')

            KNOWLEDGES_DIR = os.path.join('experiments', percentage, iteration,  "knowledges")

            SEGMENTATION_DIRECTORY = os.path.join(DATA_PATH, "segmentation")

            PR_SEG_DIRECTORY = os.path.join(DATA_PATH, "proposal_segmentations")
            if os.path.isdir(PR_SEG_DIRECTORY) == False:
                os.mkdir(PR_SEG_DIRECTORY)

            PR_IMG_DIRECTORY = os.path.join(DATA_PATH, "proposal_images")
            if os.path.isdir(PR_IMG_DIRECTORY) == False:
                os.mkdir(PR_IMG_DIRECTORY)

            # Configuration des parametres

            alpha = 0.5

            nb_classes = 8

            nodes_specifier = [
                CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier()
                #MaxDistanceSpecifier.MaxDistandeSpecifier()
            ]

            node_knowledge = load_knowledge(KNOWLEDGES_DIR, 'nodes', nodes_specifier)

            edges_specifier = [
                #RelativePositionSpecifier.RelativePositionSpecifier()
                MinMaxEdtDistanceSpecifier.MinMaxEdtDistandeSpecifier()
            ]

            edge_knowledge = load_knowledge(KNOWLEDGES_DIR, 'edges', edges_specifier)

            max_node_matching = 2
            max_node_refinement = math.inf

            # Methodes

            files = os.listdir(SEGMENTATION_DIRECTORY)

            prog_bar = bar.Bar("Processing {} - percentage {} - iteration {}".format(architecture, percentage, iteration), max=len(files))
            for k in files:

                filename = k.replace('.npy', '')

                result_path = os.path.join(PR_SEG_DIRECTORY, "{}_result.csv".format(filename))

                if os.path.isfile(result_path) is False:

                    rgb_name = k.replace('.npy', '.png')
                    label_name = rgb_name.replace('npy', '.png')
                    
                    """""""""""""""""""""""""""""""""""""""
                        Chargement des données
                    """""""""""""""""""""""""""""""""""""""

                    pr_mask = np.load(os.path.join(SEGMENTATION_DIRECTORY, k))

                    if architecture == 'U_Net_CRF' or architecture == 'U_Net':
                        pr_mask = tensor(pr_mask)
                        pr_mask = softmax(pr_mask, dim=0)
                        pr_mask = pr_mask.numpy()

                        pr_mask = np.transpose(pr_mask, (1,2,0))

                    image_cnn = np.argmax(pr_mask, axis=2)

                    dims = len(image_cnn.shape)
                    
                    """""""""""""""""""""""""""""""""""""""
                        Etapes de pre-traitements
                    """""""""""""""""""""""""""""""""""""""

                    image_cnn = median_filter(image_cnn, (3,3))

                    present_values = np.nonzero(np.unique(image_cnn))[0]

                    if len(present_values) == 8:
                    
                        params = {}
                        
                        params['Cs'] = get_diagonal_size(image_cnn)
                        params['weigthed'] = True
                        params['lbd'] = 0.5
                        params['min_max_coef'] = 0.5

                        pr_mask = pr_mask[:,:, 1:nb_classes + 1]

                        """""""""""""""""""""""""""""""""""""""
                            Initial matching
                        """""""""""""""""""""""""""""""""""""""

                    
                        labelled_image = label(image_cnn)
                        regions = regionprops(labelled_image, image_cnn)

                        M = define_permutations(regions, nb_classes, max_node_matching)
                        
                        kv_constructor = KvConstructor(nodes_specifier, [1], node_knowledge)

                        kv = kv_constructor.construct_Kv_initial(pr_mask, labelled_image, regions, params, nb_classes)

                        ke_constructor = KeConstructor(edges_specifier, [1], edge_knowledge)
                            
                        ke = ke_constructor.construct_Ke_initial(pr_mask, labelled_image, regions, params, nb_classes)

                        best_permutation = None
                        best_score_perm = math.inf
                        for permutation in M:

                            score = evaluate_permutation(permutation, kv, ke)
                            
                            if score < best_score_perm:
                                best_score_perm = score
                                best_permutation = permutation
                        
                        # verif

                        best_matching = permutation_to_matching(best_permutation)
                        
                        kv_constructor = KvConstructor(nodes_specifier, [1], node_knowledge)
                        
                        kv = kv_constructor.construct_Kv(pr_mask, labelled_image, regions, best_matching, params)
                        
                        ke_constructor = KeConstructor(edges_specifier, [1], edge_knowledge)
                        
                        ke = ke_constructor.construct_Ke(pr_mask, labelled_image, regions, best_matching, params)

                        best_score = evaluate_matching(best_matching, kv, ke)

                        matching_image = create_images_from_ids(labelled_image, best_matching)

                        Aes = ke_constructor.Aes
                        
                            
                        """""""""""""""""""""""""""""""""""""""
                            Refinement
                        """""""""""""""""""""""""""""""""""""""
                        
                        list_ids = list(np.unique(labelled_image)[1:] - 1)
                        
                        for k,v in best_matching.items():
                            for node in v:
                                list_ids.remove(node)
                        
                        temp_best_matching = deepcopy(best_matching)
                        temp_best_score = best_score
                        temp_Ans = deepcopy(kv_constructor.Ans)
                        temp_Aes = deepcopy(ke_constructor.Aes)
                        temp_ke = deepcopy(ke)
                        
                        for ids in list_ids:

                            class_best_matching = deepcopy(temp_best_matching)
                            class_best_score = math.inf
                            class_best_Ans = deepcopy(temp_Ans)
                            class_best_Aes = deepcopy(temp_Aes)
                            class_best_ke = deepcopy(temp_ke)
                            
                            for cls in range(0, nb_classes):

                                temp_matching = deepcopy(temp_best_matching)
                                actual_ke = deepcopy(temp_ke)
                                
                                temp_matching[cls].append(ids)
                                
                                kv_constructor = KvConstructor(nodes_specifier, [1], node_knowledge)

                                kv_constructor.Ans = class_best_Ans
                            
                                kv = kv_constructor.construct_Kv(pr_mask, labelled_image, regions, temp_matching, params, cls)
                                
                                ke_constructor = KeConstructor(edges_specifier, [1.0], edge_knowledge)

                                ke_constructor.Aes = class_best_Aes
                                
                                ke = ke_constructor.construct_Ke(pr_mask, labelled_image, regions, temp_matching, params, cls, actual_ke)
                                
                                score = evaluate_matching(temp_matching, kv, ke)
                                
                                if score < class_best_score:
                                    class_best_score = score
                                    class_best_matching = temp_matching
                                    class_best_Ans = kv_constructor.Ans
                                    class_best_Aes = ke_constructor.Aes
                                    class_best_ke = ke
                            
                            if class_best_score < temp_best_score * 1.01:
                                temp_best_score = class_best_score
                                temp_best_matching = class_best_matching
                                temp_Ans = class_best_Ans
                                temp_Aes = class_best_Aes
                                temp_ke = class_best_ke

                        proposal_image = create_images_from_ids(labelled_image, temp_best_matching)

                        proposal_img_path = os.path.join(PR_IMG_DIRECTORY, "{}.png".format(filename))
                        proposal_seg_path = os.path.join(PR_SEG_DIRECTORY, "{}.png".format(filename))
                        
                        plt.imshow(proposal_image); plt.tight_layout(); plt.savefig(proposal_img_path); plt.close()

                        img = Image.fromarray(proposal_image).convert('L')
                        img.save(proposal_seg_path)

                prog_bar.next()
            prog_bar.finish()