import numpy as np
from src import utils

class refinement_Ar_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def construct_Ar(self, specifier, labelled_image, regions, matching, nb_classes):
        return self.constructor(specifier, labelled_image, regions, matching, nb_classes)

class Functor(object):
            
    def __call__(self, specifier, labelled_image, regions, matching, nb_classes) :

        if specifier == "centroid":
            return self. __Centroid(labelled_image, regions, matching, nb_classes)
        elif specifier == "edt_signed":
            return self. __EDT_min(labelled_image, regions, matching, nb_classes)
        else :
            return self.__Error()

    def __Centroid(self, labelled_image, regions, matching, nb_classes):
        Ar = np.zeros((nb_classes, nb_classes, len(labelled_image.shape)))

        for i in range(0, nb_classes):
            ids1 = matching[i][0]
            centro1 = regions[ids1 - 1].centroid

            for j in range(i+1, nb_classes):
                ids2 = matching[j][0]
                centro2 = regions[ids2 - 1].centroid
                
                vector = np.asarray(centro2) - np.asarray(centro1)
                vector = np.flip(vector, axis= 0)

                for dim in range(0,len(labelled_image.shape)):
                    Ar[i][j][dim] = vector[dim]
                    Ar[j][i][dim] = -vector[dim]

        Ar = np.transpose(Ar, (2,0,1))
        return Ar

    def __EDT_min(self, labelled_image, regions, matching, nb_classes):
        Ar = np.zeros((2, nb_classes, nb_classes))

        for i in range(0, nb_classes):
            ids1 = matching[i][0] - 1
            mask = np.where(labelled_image == regions[ids1].label, 1, 0)
            dist = utils.signed_transform(mask)
            
            for j in range(0, nb_classes):
                if i !=j:
                    ids2 = matching[j][0] - 1
                    mask2 = np.where(labelled_image == regions[ids2].label, 1, 0)
                    res = mask2 * dist

                    Ar[0][i][j] = np.min(res[np.nonzero(res)])
                    Ar[1][i][j] = np.max(res[np.nonzero(res)])

        #Ar = np.transpose(Ar, (2,0,1))
        return Ar
        
    def __Error(self):
        print ("Data is not correct")