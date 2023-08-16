from .EdgeSpecifier import EdgeSpecifier

import numpy as np
import math
import edt

from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes


def signed_transform(mask):

    mask = mask.astype(bool)

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

        Ae = np.zeros((nb_regions, nb_regions,nb_dims))

        for i in range(0, nb_regions):
            
            mask = np.zeros(labelled_image.shape)
            
            region1 = regions[i]

            mask = np.where(region1.label == labelled_image, 1, mask)

            dist = signed_transform(mask)
            
            for j in range(0, nb_regions):
                if i !=j:
                    
                    mask2 = np.zeros(labelled_image.shape)
            
                    region2 = regions[j]
                    mask2 = np.where(region2.label == labelled_image, 1, mask2)

                    res = mask2 * dist

                    Ae[i][j][0] = np.min(res[np.nonzero(res)])
                    Ae[i][j][1] = np.max(res[np.nonzero(res)])
        return Ae
    
    def evaluation_metrics(self, value1, value2, params):
        
        error_distance_min = math.fabs(value1[0] - value2[0]) / params['Cs']
        error_distance_max = math.fabs(value1[1] - value2[1]) / params['Cs']        
        
        error_distance = params["min_max_coef"] * error_distance_min + (1 - params["min_max_coef"]) * error_distance_max

        return error_distance