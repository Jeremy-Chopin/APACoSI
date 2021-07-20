import numpy as np
import copy
import edt
from progress.bar import Bar
from scipy.sparse import construct
from sklearn.metrics import mean_squared_error
from skimage.measure import label

from src import utils #, refinement_An_constructor, refinement_Ar_constructor, Ke_constructor, Kv_constructor
from src.Ke_constructor import Ke_constructor




class refinement_strategy_constructor(object):
    def __init__(self):
        self.constructor=Functor()
    
    def refinement(self, specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha):
        return self.constructor(specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha)

class Functor(object):
            
    def __call__(self, specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha):

        if isinstance(specifier, str):
            return self.__refinement(specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha)
        else:
            raise Exception("Specifier is not in a good format!")

    def __refinement(self, specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha):
        Ar_final = copy.deepcopy(Ar_initial)
        An_final = copy.deepcopy(An_initial)
        final_matching = copy.deepcopy(matching)
        score_final = copy.deepcopy(initial_score)

        bar = Bar("Refinement : ",max=len(list_regions_ids))
        for region_ids in list_regions_ids:
            best_merging = None
            best_score = np.inf
            best_Ar = None
            best_An = None

            for label1 in matching.keys():
        
                Ar_inter = copy.deepcopy(Ar_final)

                matching_inter = copy.deepcopy(final_matching)
                matching_inter[label1].append(region_ids)

                if specifier == "centroid":
                    Ar_inter = self.__Centroid(regions, matching_inter, label1, Ar_inter)
                elif specifier == "edt_min":
                    self.__EDT( labelled_image, regions, matching_inter, label1, Ar_inter)
                else:
                    raise Exception("Specifier is not implemented!")

                An_inter = utils.update_An(An_final, matching_inter, pr_mask, regions, labelled_image, label1, nb_classes)

                Ke = Ke_constructor().construct_Ke(specifier, Am, Ar_inter, params)
                Kv = utils.construct_Kv(An_inter)

                score = utils.calculate_matching_cost(alpha, Kv, Ke, nb_classes)

                if score < best_score:
                    best_score = score
                    best_merging = matching_inter
                    best_Ar = Ar_inter
                    best_An = An_inter

            if best_score < score_final:
                    Ar_final = best_Ar
                    An_final = best_An
                    final_matching = best_merging
                    score_final = best_score
                    
            bar.next()
        bar.update()
        print("\n")

        return final_matching

    def __Centroid(self, regions, matching_inter, label1, Ar_inter):

        centroid = []
        areas = []
        
        for ids in matching_inter[label1]:

            centroid.append(regions[ids-1].centroid)
            areas.append(regions[ids-1].area)

        centro = None
        for v in range(0, len(centroid)):
            if v == 0:
                centro = np.asarray(centroid[v]) * areas[v]
            else:
                centro +=  np.asarray(centroid[v]) * areas[v]
        zc1, yc1, xc1 = centro / sum(areas)

        for label2 in matching_inter.keys():
            if label1 != label2:
                
                centroid = []
                areas = []
                
                for ids in matching_inter[label2]:

                    centroid.append(regions[ids-1].centroid)
                    areas.append(regions[ids-1].area)

                centro = None
                for v in range(0, len(centroid)):
                    if v == 0:
                        centro = np.asarray(centroid[v]) * areas[v]
                    else:
                        centro +=  np.asarray(centroid[v]) * areas[v]
                zc2, yc2, xc2 = centro / sum(areas)

                vector = np.asarray([xc2 - xc1, yc2 - yc1, zc2 - zc1])

                for dim in range(0,3):
                    Ar_inter[dim][label1][label2] = vector[dim]
                    Ar_inter[dim][label2][label1] = -vector[dim]
        
        return Ar_inter
        
    def __EDT(self, labelled_image, regions, matching_inter, label1, Ar_inter):

        mask = np.ones(labelled_image.shape)

        for ids in matching_inter[label1]:
            mask = np.where(labelled_image == regions[ids - 1].label, 0, mask)

        unique = np.unique(label(mask, connectivity=2))

        if np.max(unique) > 1:
            mask = np.where(mask == 1, 0, 1)
            dist = utils.signed_transform(mask)
        else:
            dist = edt.edt(mask.astype(np.bool))

        for label2 in range(label1 + 1, len(matching_inter.keys())):
            mask2 = np.zeros(labelled_image.shape)

            for ids in matching_inter[label2]:
                mask2 = np.where(labelled_image == regions[ids - 1].label, 1, mask2)

            res = dist  * mask2

            min_value = np.min(res[np.nonzero(res)])

            if np.max(unique) > 1:
                max_value = utils.get_max_EDT_signed(labelled_image, regions, matching_inter[label1], matching_inter[label2])
            else:
                max_value = utils.get_max_EDT(labelled_image, regions, matching_inter[label1], matching_inter[label2])

            Ar_inter[0][label1][label2] = min_value
            Ar_inter[1][label1][label2] = max_value

            Ar_inter[0][label2][label1] = min_value
            Ar_inter[1][label2][label1] = max_value
        
        return Ar_inter
