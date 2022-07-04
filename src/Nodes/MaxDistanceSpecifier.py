from .NodeSpecifier import NodeSpecifier

import numpy as np
import math
from skimage.measure import regionprops, label
from scipy.spatial.distance import pdist
import itertools
import cc3d

def get_hdistance(mask):
    
    if len(mask.shape) == 2:
        lbl = label(mask)
    else:
        lbl = cc3d.connected_components(mask)
    
    regions = regionprops(lbl)
    
    all_coords = []

    for region in regions:
        for coord in region.coords:
            all_coords.append(coord)

    d = pdist(all_coords)

    max_d = max(d)
    
    return max_d

class MaxDistandeSpecifier(NodeSpecifier):
    
    def __init__(self):
        self.name = 'max_distance'
    
    def define_Ar(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)

        An = np.zeros((nb_classes, nb_classes))

        for i in range(0, nb_classes):
            
            img = np.zeros(labelled_image.shape)
            
            nodes = matching[i]
            for n in nodes:
                img = np.where(n + 1 == labelled_image, 1, img)
            
            #region = regionprops(img.astype(np.uint8))[0]
            #dist = region.major_axis_length
            
            dist = get_hdistance(img)
            
            An[i][i] = dist

        return An
    
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
    
    def evaluation_metrics(self, value1, value2, params):
        
        error_distance = math.fabs(value1 - value2) / params['Cs']
        
        return error_distance