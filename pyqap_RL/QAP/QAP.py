import numpy as np
import copy
import itertools
import time

from ..metrics import dice_score
from numba import jit, njit

def find_best_matching(class_to_match, assignement_matrix, K):

    #potential_matchings, regions_matched = create_all_matchings(assignement_matrix, class_to_match)
    
    nb_rows = assignement_matrix.shape[0]

    best_score = np.inf
    best_matching = None
    best_regions = None
    
    if type(class_to_match) is not tuple :
        for i in range(0, nb_rows):
            if np.sum(assignement_matrix[i,:]) == 0:
                temp = copy.deepcopy(assignement_matrix)
                temp[i][class_to_match] = 1
                
                vec_x = temp.flatten('F')
                x_translate = np.transpose(vec_x)

                score = np.dot(np.matmul(x_translate,K), vec_x)

                if score < best_score:
                    best_score = score
                    best_matching = temp
                    best_regions = i
    else:
        for iter in itertools.permutations(range(nb_rows), len(class_to_match)):
            temp = copy.deepcopy(assignement_matrix)
            for i in range(0, len(iter)):
                temp[iter[i]][class_to_match[i]] = 1

            vec_x = temp.flatten('F')
            x_translate = np.transpose(vec_x)

            score = np.dot(np.matmul(x_translate,K), vec_x)

            if score < best_score:
                best_score = score
                best_matching = temp
                best_regions = iter
    
    '''i = 0
    for matching in potential_matchings:
        vec_x = matching.flatten('F')
        x_translate = np.transpose(vec_x)

        score = np.dot(np.matmul(x_translate,K), vec_x)

        if score < best_score:
            best_score = score
            best_matching = matching
            best_regions = regions_matched[i]
        i +=1'''
    
    if best_matching is None or best_regions is None:
        print('error')
    return best_matching, best_regions

def create_all_matchings(Am, class_to_match):

    nb_rows = Am.shape[0]

    potential_matchings = []
    regions_matched = []

    if type(class_to_match) is not tuple :
        for i in range(0, nb_rows):
            if np.sum(Am[i,:]) == 0:
                temp = copy.deepcopy(Am)
                temp[i][class_to_match] = 1
                potential_matchings.append(temp)
                regions_matched.append(i)
    else:
        for iter in itertools.permutations(range(nb_rows), len(class_to_match)):
            temp = copy.deepcopy(Am)
            for i in range(0, len(iter)):
                temp[iter[i]][class_to_match[i]] = 1
            potential_matchings.append(temp)
            regions_matched.append(iter)
    
    return potential_matchings, regions_matched

"""@njit
def reward_from_dice_score(class_to_evaluate, region_to_evaluate, segmentation, gt):

    reward = 0
    #if type(class_to_evaluate) is tuple:
    if class_to_evaluate.size > 1:
        
        for i in range(0, class_to_evaluate.size):
            mask_seg = np.where(segmentation == region_to_evaluate[i]+1, True, False)
            mask_gt = np.where(gt == class_to_evaluate[i]+1, True, False)
            reward += dice_score(mask_seg, mask_gt)
        reward = reward / class_to_evaluate.size
        
    else:
        mask_seg = np.where(segmentation == region_to_evaluate.max() + 1, 1, 0)
        mask_gt = np.where(gt == class_to_evaluate.max() + 1, True, False)
        reward = dice_score(mask_seg, mask_gt)

    return reward"""

def reward_from_dice_score(class_to_evaluate, region_to_evaluate, img1, img2):

    if class_to_evaluate.size > 1:
        return reward_from_dice_score_tuple(class_to_evaluate, region_to_evaluate, img1, img2)
        
    else:
        return reward_from_dice_score_single(class_to_evaluate, region_to_evaluate, img1, img2)
 
def reward_from_dice_score_tuple(class_to_evaluate, region_to_evaluate, img1, img2):

    reward = 0
        
    for i in range(0, class_to_evaluate.size):
        mask_seg = np.where(img1 == region_to_evaluate[i]+1, True, False)
        mask_gt = np.where(img2 == class_to_evaluate[i]+1, True, False)
        reward += dice_score(mask_seg, mask_gt)
    reward = reward / class_to_evaluate.size
    return reward

   
def reward_from_dice_score_single(class_to_evaluate, region_to_evaluate, img1, img2):

    mask_seg = np.where(img1 == np.max(region_to_evaluate) + 1, 1, 0)
    mask_gt = np.where(img2 == np.max(class_to_evaluate) + 1, True, False)
    return dice_score(mask_seg, mask_gt)