import numpy as np
from sklearn.metrics import mean_squared_error

class refinement_An_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_An(self, specifier, labelled_image, regions, matching, nb_classes, pr_mask):
        return self.constructor(specifier, labelled_image, regions, matching, nb_classes, pr_mask)

class Functor(object):
            
    def __call__(self, specifier, labelled_image, regions, matching, nb_classes, pr_mask) :

        if specifier == "centroid":
            return self. __Centroid(labelled_image, regions, matching, nb_classes, pr_mask)
        elif specifier == "edt_signed":
            return self. __EDT_min(labelled_image, regions, matching, nb_classes, pr_mask)
        else :
            return self.__Error()

    def __Centroid(self, image_labelled, regions, matching, nb_classes, pr_mask):
        nb_classes = len(matching.keys())

        An = np.zeros((nb_classes, nb_classes))

        for i in range(0, nb_classes):

            region_mask = np.zeros(image_labelled.shape)

            for ids in matching[i]:
                region = regions[ids - 1]
                region_mask = np.where(image_labelled == region.label, 1, region_mask)

            proba_mask = np.expand_dims(region_mask, axis=3) * pr_mask

            region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)

            for j in range(0, nb_classes):

                vector = np.zeros(nb_classes)
                vector[j] = 1

                value = mean_squared_error(vector, region_probs)

                An[i][j] = value

        return An      

    def __EDT_min(self, labelled_image, regions, matching, nb_classes, pr_mask):
        nb_classes = len(matching.keys())

        An = np.zeros((nb_classes, nb_classes))

        for i in range(0, nb_classes):

            region_mask = np.zeros(labelled_image.shape)

            for ids in matching[i]:
                region = regions[ids - 1]
                region_mask = np.where(labelled_image == region.label, 1, region_mask)

            proba_mask = np.expand_dims(region_mask, axis=3) * pr_mask

            region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)

            for j in range(0, nb_classes):

                vector = np.zeros(nb_classes)
                vector[j] = 1

                value = mean_squared_error(vector, region_probs)

                An[i][j] = value

        return An      
    def __Error(self):
        print ("Data is not correct")