from .EdgeSpecifier import EdgeSpecifier

import numpy as np
import math
import edt

from copy import deepcopy
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes
import time
from matplotlib import pyplot as plt

def signed_transform(mask):

    mask = mask.astype(np.bool)

    negatif = np.where(mask == True, False, True)
    
    exterior_value = edt.edt(negatif)

    mask_filled = binary_fill_holes(mask)

    negatif_filled = np.where(mask_filled == True, False, True)

    interior_value = edt.edt(negatif_filled)

    complete_mask = np.where(interior_value != exterior_value, -exterior_value, exterior_value)

    return complete_mask

class MinMaxEdtDistandeSpecifier(EdgeSpecifier):
    
    def __init__(self):
        self.name = 'min_max_edt_distance'
    
    def define_Ae(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)

        Ae = np.zeros((nb_classes, nb_classes,2))

        for i in range(0, nb_classes):
            
            mask = np.zeros(labelled_image.shape)
            
            nodes = matching[i]
            for n in nodes:
                mask = np.where(n + 1 == labelled_image, 1, mask)

            dist = signed_transform(mask)
            
            for j in range(0, nb_classes):
                if i !=j:
                    
                    mask2 = np.zeros(labelled_image.shape)
            
                    nodes = matching[j]
                    for n in nodes:
                        mask2 = np.where(n + 1 == labelled_image, 1, mask2)
                    res = mask2 * dist

                    Ae[i][j][0] = np.min(res[np.nonzero(res)])
                    Ae[i][j][1] = np.max(res[np.nonzero(res)])

        return Ae
    
    def define_Ae_knowledge(self, annotation, params):
        nb_classes = len(np.unique(annotation)) - 1 

        Ae = np.zeros((nb_classes, nb_classes, 2))

        for i in range(0, nb_classes):
            mask = np.where(annotation == i+1, 1, 0)
            dist = signed_transform(mask)
            
            for j in range(0, nb_classes):
                if i !=j:
                    mask2 = np.where(annotation == j+1, 1, 0)
                    res = mask2 * dist

                    Ae[i][j][0]= np.min(res[np.nonzero(res)])
                    Ae[i][j][1] = np.max(res[np.nonzero(res)])

        return Ae

    def define_Ae_initial(self, segmentation_map, labelled_image, regions, params, nb_classes):
        nb_regions = len(regions)
        
        nb_dims = len(labelled_image.shape)

        Ae = np.zeros((nb_regions, nb_regions, nb_dims))

        for i in range(0, nb_regions):
            mask = np.where(labelled_image == i+1, 1, 0)
            dist = signed_transform(mask)

            for j in range(0, nb_regions):
                if i != j:
                    mask2 = np.where(labelled_image == j+1, 1, 0)
                    res = mask2 * dist
                    
                    Ae[i][j][0]= np.min(res[np.nonzero(res)])
                    Ae[i][j][1] = np.max(res[np.nonzero(res)])

        return Ae

    def define_Ae_refinement(self, segmentation_map, labelled_image, regions, matching, params, label_to_update, Ae):

        #t0 = time.perf_counter()
        nb_classes = len(matching)
        
        nb_dims = len(labelled_image.shape)

        temp_Ae = deepcopy(Ae)

        mask1 = np.zeros(labelled_image.shape, dtype=np.uint8)
            
        nodes = matching[label_to_update]
        for n in nodes:
            mask1 = np.where(regions[n].label == labelled_image, 1, mask1)
            
        dist1 = signed_transform(mask1)

        for i in range(0, nb_classes):
            if i != label_to_update:
                mask2 = np.zeros(labelled_image.shape, dtype=np.uint8)
                
                nodes = matching[i]
                for n in nodes:
                    mask2 = np.where(regions[n].label == labelled_image, 1, mask2)
                dist2 = signed_transform(mask2)

                # i vers update
                res1 = mask2 * dist1
                
                try:
                    temp_Ae[i][label_to_update][0]= np.min(res1[np.nonzero(res1)])
                    temp_Ae[i][i][1] = np.max(res1[np.nonzero(res1)])
                except:
                    plt.subplot(2,2,1); plt.imshow(mask1)
                    plt.subplot(2,2,2); plt.imshow(dist1)
                    plt.subplot(2,2,3); plt.imshow(mask2)
                    plt.subplot(2,2,4); plt.imshow(res1)
                    plt.show()

                # update vers i
                res2 = mask1 * dist2

                temp_Ae[label_to_update][i][0]= np.min(res2[np.nonzero(res2)])
                temp_Ae[label_to_update][label_to_update][1] = np.max(res2[np.nonzero(res2)])

        #print("Ae : ", time.perf_counter() - t0)

        return temp_Ae
    
    def evaluation_metrics(self, value1, value2, params):
        
        error_distance_min = math.fabs(value1[0] - value2[0]) / params['Cs']
        error_distance_max = math.fabs(value2[0] - value2[1]) / params['Cs']        
        
        error_distance = params["min_max_coef"] * error_distance_min + (1 - params["min_max_coef"]) * error_distance_max

        return error_distance