from cProfile import label
from .NodeSpecifier import NodeSpecifier

import numpy as np
import math
from skimage.measure import regionprops
from sklearn.metrics import mean_squared_error
from copy import deepcopy


class CnnProbabilitiesSpecifier(NodeSpecifier):
    
    def __init__(self):
        self.name = 'cnn_probabilities'
    
    def define_Ar(self, segmentation_map, labelled_image, regions, matching, params):
        nb_classes = len(matching)

        An = np.zeros((nb_classes, nb_classes, nb_classes))

        for i in range(0, nb_classes):

            region_mask = np.zeros(labelled_image.shape)

            nodes = matching[i]
            for n in nodes:
                region_mask = np.where(n + 1 == labelled_image, 1, region_mask)

            if len(labelled_image.shape) == 2:
                proba_mask = np.expand_dims(region_mask, axis=2) * segmentation_map
                region_probs = np.sum(proba_mask, axis=(0,1)) / np.sum(region_mask)
            else:
                proba_mask = np.expand_dims(region_mask, axis=3) * segmentation_map
                region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)

            for k in range(0, len(region_probs)):
                An[i][i][k] = region_probs[k]

        return An

    def define_Ar_refinement(self, segmentation_map, labelled_image, regions, matching, params, label_to_update, An):
        nb_classes = len(matching)

        temp_An = deepcopy(An)

        region_mask = np.zeros(labelled_image.shape)

        nodes = matching[label_to_update]
        for n in nodes:
            region_mask = np.where(n + 1 == labelled_image, 1, region_mask)

        if len(labelled_image.shape) == 2:
            proba_mask = np.expand_dims(region_mask, axis=2) * segmentation_map
            region_probs = np.sum(proba_mask, axis=(0,1)) / np.sum(region_mask)
        else:
            proba_mask = np.expand_dims(region_mask, axis=3) * segmentation_map
            region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)

        for k in range(0, len(region_probs)):
            temp_An[label_to_update][label_to_update][k] = region_probs[k]

        return temp_An

    def define_Ar_knowledge(self, annotation):
        nb_classes = len(np.unique(annotation)) - 1 

        An = np.zeros((nb_classes, nb_classes, nb_classes))

        for i in range(0, nb_classes):
            
            An[i][i][i] = 1

        return An

    def define_Ar_initial(self, segmentation_map, labelled_image, regions, params, nb_classes):
        nb_regions = len(regions)

        An = np.zeros((nb_regions, nb_regions, nb_classes))

        for i in range(0, nb_regions):

            region_mask = np.zeros(labelled_image.shape)

            region = regions[i]

            region_mask = np.where(region.label == labelled_image, 1, region_mask)

            if len(labelled_image.shape) == 2:
                proba_mask = np.expand_dims(region_mask, axis=2) * segmentation_map
                region_probs = np.sum(proba_mask, axis=(0,1)) / np.sum(region_mask)
            else:
                proba_mask = np.expand_dims(region_mask, axis=3) * segmentation_map
                region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)

            for k in range(0, len(region_probs)):
                An[i][i][k] = region_probs[k]

        return An 
    
    def evaluation_metrics(self, value1, value2, params):
        
        error_probs = mean_squared_error(value1, value2)
        
        return error_probs