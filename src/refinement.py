import numpy as np
import os 
import pandas
import math
from scipy import linalg as LA
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
import copy
from progress.bar import Bar
import copy
from src import utils
from skimage.measure import label
import edt
from src.refinement_Ar_constructor import refinement_Ar_constructor
from src.refinement_An_constructor import refinement_An_constructor
from src.Ke_constructor import Ke_constructor
from src.Kv_constructor import Kv_constructor
import time

def change_variable_and_check(specifier, nb_classes, labelled_image, best_matching, best_cost, Am, alpha, pr_mask, params, nodes_matching, regions):
    matching = {}

    for i in range(0, nb_classes):
        matching[i] = []

    x,y  = np.where(best_matching == 1)

    for i in range(0,len(x)):
        label = y[i]
        pos = x[i]
        matching[label].append(nodes_matching[pos].ids)

    Ar = refinement_Ar_constructor().construct_Ar(specifier, labelled_image, regions, matching, nb_classes)
    An = refinement_An_constructor().construct_An(specifier, labelled_image, regions, matching, nb_classes, pr_mask)

    # Etape 2 : Vérification - tests de non-regression

    Ke = Ke_constructor().construct_Ke(specifier, Am, Ar, params)
    Kv = __construct_Kv(An)

    score_non_regression = __calculate_matching_cost(alpha, Kv, Ke, matching, nb_classes)

    if np.round(score_non_regression, 5) != np.round(best_cost, 5):
        print("Erreur on image : " + str(k) + " !!!") 

    return score_non_regression, Ar, An, matching

def __construct_Kv(An):
        
    nb_classes = An.shape[0]
    Kv = np.zeros((nb_classes * nb_classes, nb_classes * nb_classes))

    for i in range(0, nb_classes * nb_classes):
        row = int(i / nb_classes)
        col = i % nb_classes
        Kv[i][i] = An[row][col]

    return Kv

def __calculate_matching_cost(alpha, Kv, Ke, matching, nb_classes):
    
    X = np.zeros((nb_classes, nb_classes))
    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i == j:
                X[i][j] = 1

    x,y = X.shape

    vec_x = X.flatten('F')

    x_translate = np.transpose(vec_x)

    tempo_ke = np.matmul(x_translate,Ke)
    score_Ke = np.dot(tempo_ke,vec_x)

    tempo_kv = np.matmul(x_translate,Kv)
    score_Kv = np.dot(tempo_kv,vec_x)

    return (1-alpha)* score_Ke + alpha * score_Kv

def apply_refinement_brain(specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha):

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

        for label in matching.keys():
    
            Ar_inter = copy.deepcopy(Ar_final)

            matching_inter = copy.deepcopy(final_matching)
            matching_inter[label].append(region_ids)

            centroid = []
            areas = []
            
            for ids in matching_inter[label]:

                centroid.append(regions[ids-1].centroid)
                areas.append(regions[ids-1].area)

            centro = None
            for v in range(0, len(centroid)):
                if v == 0:
                    centro = np.asarray(centroid[v]) * areas[v]
                else:
                    centro +=  np.asarray(centroid[v]) * areas[v]
            zc1, yc1, xc1 = centro / sum(areas)

            for label2 in matching.keys():
                if label != label2:
                    
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
                        Ar_inter[dim][label][label2] = vector[dim]
                        Ar_inter[dim][label2][label] = -vector[dim]
            
            An_inter = __update_An(An_final, matching_inter, pr_mask, regions, labelled_image, label, nb_classes)

            Ke = Ke_constructor().construct_Ke(specifier, Am, Ar_inter, params)
            Kv = __construct_Kv(An_inter)

            score = __calculate_matching_cost(alpha, Kv, Ke, matching, nb_classes)

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


def apply_refinement_face(specifier, labelled_image, regions, matching, Am, Ar_initial, An_initial, list_regions_ids, initial_score, pr_mask, nb_classes, params, alpha):
    
    Ar_final = copy.deepcopy(Ar_initial)
    An_final = copy.deepcopy(An_initial)
    final_matching = copy.deepcopy(matching)
    score_final = copy.deepcopy(initial_score)
    

    bar = Bar("Refinement : ",max=len(list_regions_ids))
    for region_ids in list_regions_ids:
        tinitial = time.time()
        best_merging = None
        best_score = np.inf
        best_Ar = None
        best_An = None

        for label1 in matching.keys():
            t_label = time.time()
            Ar_inter = copy.deepcopy(Ar_initial)

            matching_inter = copy.deepcopy(final_matching)
            matching_inter[label1].append(region_ids)

            mask = np.ones(labelled_image.shape)

            for ids in matching_inter[label1]:
                mask = np.where(labelled_image == regions[ids - 1].label, 0, mask)

            unique = np.unique(label(mask, connectivity=2))

            if np.max(unique) > 1:
                mask = np.where(mask == 1, 0, 1)
                dist = utils.signed_transform(mask)
            else:
                dist = edt.edt(mask.astype(np.bool))

            #test = label(mask, connectivity=2)
            #v = np.unique(test)

            #print("Opération 1 : ", time.time() - t2)

            # On recalcule les distances par rapports aux autres noeuds
            for label2 in range(label1 + 1, nb_classes):
                #if label1 != label2:

                    #t_label2 = time.time()
                    mask2 = np.zeros(labelled_image.shape)

                    for ids in matching_inter[label2]:
                        mask2 = np.where(labelled_image == regions[ids - 1].label, 1, mask2)

                    res = dist  * mask2

                    min_value = np.min(res[np.nonzero(res)])
                    #max_value = np.max(res[np.nonzero(res)])

                    #print("Opération 2 : ", time.time() - t3)

                    #t1 = time.time()
                    if np.max(unique) > 1:
                        max_value = utils.get_max_EDT_signed(labelled_image, regions, matching_inter[label1], matching_inter[label2])
                    else:
                        max_value = utils.get_max_EDT(labelled_image, regions, matching_inter[label1], matching_inter[label2])
                    #print("Opération 3 : ", time.time() - t1)
                    Ar_inter[0][label1][label2] = min_value
                    Ar_inter[1][label1][label2] = max_value

                    Ar_inter[0][label2][label1] = min_value
                    Ar_inter[1][label2][label1] = max_value

                    #print("Label_2 : ", time.time() - t_label2)

            An_inter = __update_An(An_final, matching_inter, pr_mask, regions, labelled_image, label1, nb_classes)

            Ke = Ke_constructor().construct_Ke(specifier, Am, Ar_inter, params)
            Kv = __construct_Kv(An_inter)

            score = __calculate_matching_cost(alpha, Kv, Ke, matching, nb_classes)

            if score < best_score:
                best_score = score
                best_merging = matching_inter
                best_Ar = Ar_inter
                best_An = An_inter
            
            print("Label_1 : ", time.time() - t_label)
        
        if best_score < score_final:
            Ar_final = best_Ar
            An_final = best_An
            final_matching = best_merging
            score_final = best_score

        print("Iteration : ", time.time() - tinitial)
            
        bar.next()
    bar.update()
    print("\n")

    return final_matching

def __update_An(An_initial, matching, pr_mask, regions, image_labelled, label_to_test, nb_classes):

    An_inter = copy.deepcopy(An_initial)
    region_mask = np.zeros(image_labelled.shape)

    for ids in matching[label_to_test]:
        region = regions[ids - 1]
        region_mask = np.where(image_labelled == region.label, 1, region_mask)

    if len(image_labelled.shape) > 2:
        proba_mask = np.expand_dims(region_mask, axis=3) * pr_mask
        region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)
    else:
        proba_mask = np.expand_dims(region_mask, axis=2) * pr_mask
        region_probs = np.sum(proba_mask, axis=(0,1)) / np.sum(region_mask)

    for j in range(0, nb_classes):

        vector = np.zeros(nb_classes)
        vector[j] = 1

        value = mean_squared_error(vector, region_probs)

        An_inter[label_to_test][j] = value

    return An_inter

def create_images_from_ids(labelled_image, list_ids, regions):
    
    image_inter = np.zeros(labelled_image.shape)

    for k,v in list_ids.items():
        for ids in v:
            image_inter = np.where(labelled_image == regions[ids - 1].label, k + 1, image_inter)
    
    return image_inter

def correct_merging_distance(image_labelled, regions, matching_initial, final_matching, distance):
    
    matching_inter = copy.deepcopy(matching_initial)
    distance_recap = {}

    for i in matching_initial.keys():
        ids1 = matching_initial[i][0]
        distance_recap[i] = []

        z1, y1, x1 = regions[ids1-1].centroid

        for ids2 in final_matching[i]:
            if ids1 != ids2:
                z2, y2, x2 = regions[ids2-1].centroid

                d = math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2) + math.pow(z2-z1, 2))

                distance_recap[i].append(d)

                if d < distance:
                    matching_inter[i].append(ids2)

    return matching_inter, distance_recap