import time

from .EdgeSpecifier import EdgeSpecifier

import numpy as np
import math
import cc3d

from numpy.linalg import norm
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass
from copy import deepcopy


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
    
    return mask_centroid
        

class RelativePositionSpecifier(EdgeSpecifier):
    
    def __init__(self):
        self.name = 'relative_position'
    
    def define_Ae(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)
        
        nb_dims = len(labelled_image.shape)

        Ae = np.zeros((nb_classes, nb_classes, nb_dims))

        for i in range(0, nb_classes):
            mask = np.zeros(labelled_image.shape, dtype=np.uint8)
            
            nodes = matching[i]
            for n in nodes:
                mask = np.where(n + 1 == labelled_image, 1, mask)
            
            centro1 = define_centroid(mask, params['weigthed'])

            for j in range(i+1, nb_classes):
                mask2 = np.zeros(labelled_image.shape, dtype=np.uint8)
            
                nodes = matching[j]
                for n in nodes:
                    mask2 = np.where(n + 1 == labelled_image, 1, mask2)
                    
                centro2 = define_centroid(mask2, params['weigthed'])
                
                vector = centro1 - centro2

                for dim in range(0,nb_dims):
                    Ae[i][j][dim] = vector[dim]
                    Ae[j][i][dim] = -vector[dim]

        return Ae

    def define_Ae_refinement(self, segmentation_map, labelled_image, regions, matching, params, label_to_update, Ae):

        t0 = time.perf_counter()
        nb_classes = len(matching)
        
        nb_dims = len(labelled_image.shape)

        temp_Ae = deepcopy(Ae)

        mask2 = np.zeros(labelled_image.shape, dtype=np.uint8)
            
        nodes = matching[label_to_update]
        for n in nodes:
            mask2 = np.where(n + 1 == labelled_image, 1, mask2)
            
        centro2 = define_centroid(mask2, params['weigthed'])

        for i in range(0, nb_classes):
            if i != label_to_update:
                mask = np.zeros(labelled_image.shape, dtype=np.uint8)
                
                nodes = matching[i]
                for n in nodes:
                    mask = np.where(n + 1 == labelled_image, 1, mask)
                
                centro1 = define_centroid(mask, params['weigthed'])
                
                vector = centro1 - centro2

                temp_Ae[i,label_to_update,:] = vector
                temp_Ae[label_to_update, i, :] = -vector
        
        print("Ae : ", time.perf_counter() - t0)

        return temp_Ae
    
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

                    Ae[i,j,:] = vector
                    Ae[j, i, :] = -vector

        return Ae

    def define_Ae_initial(self, segmentation_map, labelled_image, regions, params, nb_classes):
        nb_regions = len(regions)
        
        nb_dims = len(labelled_image.shape)

        Ae = np.zeros((nb_regions, nb_regions, nb_dims))

        for i in range(0, nb_regions):
            mask = np.zeros(labelled_image.shape, dtype=np.bool)
            
            region = regions[i]

            mask = np.where(region.label == labelled_image, True, mask)
        
            centro1 = define_centroid(mask, params['weigthed'])

            for j in range(i+1, nb_regions):
                mask2 = np.zeros(labelled_image.shape, dtype=np.uint8)
            
                region = regions[j]
                
                mask2 = np.where(region.label == labelled_image, True, mask2)
                
                centro2 = define_centroid(mask2, params['weigthed'])
                
                vector = centro1 - centro2

                
                Ae[i,j,:] = vector
                Ae[j, i, :] = -vector
                

                """for dim in range(0,nb_dims):
                    Ae[i][j][dim] = vector[dim]
                    Ae[j][i][dim] = -vector[dim]"""
                
                

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