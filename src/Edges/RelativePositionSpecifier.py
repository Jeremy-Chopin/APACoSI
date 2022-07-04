from .EdgeSpecifier import EdgeSpecifier

import numpy as np
import math
import cc3d

from numpy.linalg import norm
from skimage.measure import label, regionprops


def define_centroid(mask, weigthed = True):
    
    if len(mask.shape) ==2:
        lbl = label(mask)
    else:
        lbl = cc3d.connected_components(mask)
        
    regions = regionprops(lbl)
    
    mask_centroid = None
    
    areas = 0
    
    for region in regions:
        
        if mask_centroid is None:
            if weigthed:
                mask_centroid = np.array(region.centroid) * region.area
                areas+=region.area
            else:
                mask_centroid = np.array(region.centroid)
        else:
            if weigthed:
                mask_centroid += np.array(region.centroid) * region.area
                areas+=region.area
            else:
                mask_centroid += np.array(region.centroid)
    
    if weigthed:
        mask_centroid = mask_centroid / areas
    else:
        mask_centroid = mask_centroid / len(regions)
    
    mask_centroid = np.floor(mask_centroid)
    
    return mask_centroid
        

class RelativePositionSpecifier(EdgeSpecifier):
    
    def __init__(self):
        self.name = 'relative_position'
    
    def define_Ae(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)
        
        nb_dims = len(labelled_image.shape)

        Ae = np.zeros((nb_classes, nb_classes, nb_dims))

        for i in range(0, nb_classes):
            mask = np.zeros(labelled_image.shape)
            
            nodes = matching[i]
            for n in nodes:
                mask = np.where(n + 1 == labelled_image, 1, mask)
            
            centro1 = define_centroid(mask, params['weigthed'])

            for j in range(i+1, nb_classes):
                mask2 = np.zeros(labelled_image.shape)
            
                nodes = matching[j]
                for n in nodes:
                    mask2 = np.where(n + 1 == labelled_image, 1, mask2)
                    
                centro2 = define_centroid(mask2, params['weigthed'])
                
                vector = centro1 - centro2

                for dim in range(0,nb_dims):
                    Ae[i][j][dim] = vector[dim]
                    Ae[j][i][dim] = -vector[dim]

        return Ae
    
    def define_Ae_knowledge(self, annotation, params):
        nb_classes = len(np.unique(annotation)) - 1 
        nb_dims = len(annotation.shape)

        Ae = np.zeros((nb_classes, nb_classes, nb_dims))

        for i in range(0, nb_classes):
            mask = np.where(annotation == i+1, 1, 0)
            centro1 = define_centroid(mask, params['weigthed'])
            
            for j in range(0, nb_classes):
                if i !=j:
                    mask2 = np.where(annotation == j+1, 1, 0)
                    centro2 = define_centroid(mask2, params['weigthed'])
                    
                    vector = centro1 - centro2

                    for dim in range(0,nb_dims):
                        Ae[i][j][dim] = vector[dim]
                        Ae[j][i][dim] = -vector[dim]

        return Ae
    
    def evaluation_metrics(self, value1, value2, params):
        
        norm1 = norm(value1)
        norm2 = norm(value2)

        scalar_pro = np.dot(value1, value2)

        if norm1 !=0 and norm2 !=0:
            cos_angle = scalar_pro / ( norm1 * norm2)

            error_postition = params["lbd"] * math.fabs(cos_angle - 1)/2 + (1-params["lbd"]) * math.fabs(norm1 - norm2) / params["Cs"]

            return error_postition
        else:
            return 0