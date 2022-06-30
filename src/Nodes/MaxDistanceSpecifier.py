from .NodeSpecifier import NodeSpecifier

import numpy as np
import math
from skimage.measure import regionprops, label
from scipy.spatial.distance import pdist
from copy import deepcopy
import itertools
import cc3d

def get_hdistance(mask):
    
    if len(mask.shape) == 2:
        lbl = label(mask)
    else:
        lbl = cc3d.connected_components(mask)
    
    regions = regionprops(lbl)
    
    all_coords = []

    """for region in regions:
        for coord in region.coords:
            all_coords.append(coord)"""

    for region in regions:
        all_coords += list(region.coords)

    if len(all_coords) == 1:
        max_d = 1
    else:
        d = pdist(all_coords)
        max_d = np.max(np.array(d))
    
    return max_d

class MaxDistandeSpecifier(NodeSpecifier):
    
    def __init__(self):
        self.name = 'max_distance'
    
    def define_Ar(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)

        An = np.zeros((nb_classes, nb_classes))

        for i in range(0, nb_classes):
            
            img = np.zeros(labelled_image.shape, dtype=np.uint8)
            
            nodes = matching[i]
            for n in nodes:
                img = np.where(n + 1 == labelled_image, 1, img)
            
            #region = regionprops(img.astype(np.uint8))[0]
            #dist = region.major_axis_length
            
            dist = get_hdistance(img)
            
            An[i][i] = dist

        return An

    def define_Ar_refinement(self, segmentation_map, labelled_image, regions, matching, params, label_to_update, An):
        nb_classes = len(matching)

        temp_An = deepcopy(An)
            
        img = np.zeros(labelled_image.shape, dtype=np.uint8)
        
        nodes = matching[label_to_update]
        for n in nodes:
            img = np.where(n + 1 == labelled_image, 1, img)
        
        dist = get_hdistance(img)
        
        temp_An[label_to_update][label_to_update] = dist

        return temp_An
    
    def define_Ar_knowledge(self, annotation):
        nb_classes = len(np.unique(annotation)) - 1 

        An = np.zeros((nb_classes, nb_classes))

        for i in range(0, nb_classes):
            
            actual_class = i+1

            img = np.where(actual_class == annotation, 1, 0)
            
            #region = regionprops(img.astype(np.uint8))[0]
            #dist = region.major_axis_length
            
            dist = get_hdistance(img)
            
            An[i][i] = dist

        return An

    def define_Ar_initial(self, segmentation_map, labelled_image, regions, params, nb_classes):

        nb_regions = len(regions)

        An = np.zeros((nb_regions, nb_regions))

        for i in range(0, nb_regions):
            
            img = np.zeros(labelled_image.shape, dtype=np.uint8)
            
            region = regions[i]

            img = np.where(region.label == labelled_image, 1, img)
            
            dist = get_hdistance(img)
            
            An[i][i] = dist

        return An
    
    def evaluation_metrics(self, value1, value2, params):
        
        error_distance = math.fabs(value1 - value2) / params['Cs']
        
        return error_distance