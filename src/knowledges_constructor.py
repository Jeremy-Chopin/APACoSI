import numpy as np
from sklearn.metrics import mean_squared_error

from skimage.measure import regionprops, label
from matplotlib import pyplot as plt
from src import utils

def keep_largest_component(mask):
    
    lbl = label(mask)
    regions = regionprops(lbl)

    label_to_keep = None
    max_area = -np.inf

    for region in regions:
        if region.area > max_area:
            label_to_keep = region.label
            max_area = region.area
    
    mask = np.where(lbl == label_to_keep, 1, 0)

    return mask

class knowledges_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_knowledges(self, specifier, gt_image, nb_classes):
        return self.constructor(specifier, gt_image, nb_classes)

class Functor(object):
            
    def __call__(self, specifier, gt_image, nb_classes) :

        if specifier == "centroid":
            return self. __Centroid(gt_image, nb_classes)
        elif specifier == "edt":
            return self. __EDT_min(gt_image, nb_classes)
        else :
            return self.__Error()

    def __Centroid(self, gt_image, nb_classes):
        Ar = np.zeros((nb_classes, nb_classes, len(gt_image.shape)))

        for i in range(0, nb_classes):
            mask1 = np.where(gt_image == i+1, 1, 0)
            mask1 = keep_largest_component(mask1)
            centro1 = regionprops(mask1)[0].centroid
            for j in range(0, nb_classes):
                if i != j:
                    mask2 = np.where(gt_image == j+1, 1, 0)
                    mask2 = keep_largest_component(mask2)
                    centro2 = regionprops(mask2)[0].centroid
                    
                    diff = np.asarray(centro2) - np.asarray(centro1)

                    if len(gt_image.shape) == 2:
                        vector = np.asarray([diff[1], diff[0]])
                    else:
                        vector = np.asarray([diff[2], diff[1], diff[0]])

                    for dim in range(0, len(gt_image.shape)):
                        Ar[i][j][dim] = vector[dim]
                        Ar[j][i][dim] = -vector[dim]

        Ar = np.transpose(Ar, (2,0,1))
        return Ar    

    def __EDT_min(self, gt_image, nb_classes):
        Ar = np.zeros((2, nb_classes, nb_classes))
        
        for i in range(0, nb_classes):
            mask = np.where(gt_image == i+1, 1, 0)
            dist = utils.signed_transform(mask)
            
            for j in range(0, nb_classes):
                if i !=j:
                    mask2 = np.where(gt_image == j+1, 1, 0)
                    res = mask2 * dist

                    Ar[0][i][j] = np.min(res[np.nonzero(res)])
                    Ar[1][i][j] = np.max(res[np.nonzero(res)])
        return Ar

    def __Error(self):
        print ("Data is not correct")