from cmath import nanj
import numpy as np
from numba import njit
from sklearn.metrics import precision_score, recall_score
import itk
from multiprocessing import Pool
import SimpleITK as sitk


#@njit
def dice_score(segmentation, gt):
    
    inter = np.logical_and(segmentation, gt)

    dice_score = np.sum(inter)*2.0 / (np.sum(segmentation) + np.sum(gt))

    return dice_score

#@njit
def avg_dice_score(segmentation, gt, nb_labels = 14):
    
    dices = []
    
    for i in range(1, nb_labels+1):
        mask_seg = np.where(segmentation == i, True, 0)
        mask_gt = np.where(gt == i, True, 0)
        dices.append(dice_score(mask_seg, mask_gt))
        
    return dices

"""def avg_dice_score(segmentation, gt, details = False):
    
    dices = []
    
    for i in range(1, len(np.unique(gt))):
        mask_seg = np.where(segmentation == i, True, 0)
        mask_gt = np.where(gt == i, 1, 0)
        dices.append(dice_score(mask_seg, mask_gt))
        
    if details is False:
        dices = np.mean(dices)
        
    return dices"""

def HD(img, annotation, nb_labels = 14):
    score = np.zeros(nb_labels)

    for i in range(1, nb_labels+1):
        mask_gt = np.where(annotation == i, 1, 0)
        mask_seg = np.where(img == i, 1, 0)

        mask_gt = np.ascontiguousarray(mask_gt)
        mask_seg = np.ascontiguousarray(mask_seg)

        image_seg = itk.GetImageFromArray(mask_seg.astype(np.float32))
        image_gt = itk.GetImageFromArray(mask_gt.astype(np.float32))

        a2t = itk.DirectedHausdorffDistanceImageFilter.New(image_gt,image_seg)
        t2a = itk.DirectedHausdorffDistanceImageFilter.New(image_seg,image_gt)

        a2t.Update()
        t2a.Update()

        v1 = a2t.GetDirectedHausdorffDistance()
        v2 = t2a.GetDirectedHausdorffDistance()

        score[i-1] = max(v1, v2)
    
    #score = score[1:]
    
    return score

def HD_sitk(img, annotation, nb_labels = 14):
    score = np.zeros(nb_labels)

    for i in range(1, nb_labels+1):
        mask_gt = np.where(annotation == i, 1, 0)
        mask_seg = np.where(img == i, 1, 0)

        mask_gt = np.ascontiguousarray(mask_gt)
        mask_seg = np.ascontiguousarray(mask_seg)

        image_seg = sitk.GetImageFromArray(mask_seg.astype(np.float32))
        image_gt = sitk.GetImageFromArray(mask_gt.astype(np.float32))

        hd_filter = sitk.HausdorffDistanceImageFilter()
        
        hd_filter.Execute(image_seg, image_gt)
        
        score[i-1] = hd_filter.GetHausdorffDistance()
    
    return score

def HD_refactor(img, annotation):
    classes_labels = np.unique(annotation)

    score = np.zeros(len(classes_labels)-1)
    
    img = np.ascontiguousarray(img)
    annotation = np.ascontiguousarray(annotation)
    
    arg = []
    
    for i in range(1, len(classes_labels)):
        mask_gt = np.where(annotation == i, 1, 0).astype(np.float32)
        mask_seg = np.where(img == i, 1, 0).astype(np.float32)
        arg.append((mask_gt, mask_seg))
    
    p = Pool()
    score = p.starmap(calculate_hd, arg)

    """for i in range(1, len(classes_labels)):
        mask_gt = np.where(annotation == i, 1, 0).astype(np.float32)
        mask_seg = np.where(img == i, 1, 0).astype(np.float32)

        image_seg = itk.GetImageFromArray(mask_seg)
        image_gt = itk.GetImageFromArray(mask_gt)

        a2t = itk.DirectedHausdorffDistanceImageFilter.New(image_gt,image_seg)
        t2a = itk.DirectedHausdorffDistanceImageFilter.New(image_seg,image_gt)

        a2t.Update()
        t2a.Update()

        v1 = a2t.GetDirectedHausdorffDistance()
        v2 = t2a.GetDirectedHausdorffDistance()

        score[i-1] = max(v1, v2)"""
    
    #score = score[1:]
    
    return score

def calculate_hd(mask_gt, mask_seg):
    
    image_seg = itk.GetImageFromArray(mask_seg)
    image_gt = itk.GetImageFromArray(mask_gt)

    a2t = itk.DirectedHausdorffDistanceImageFilter.New(image_gt,image_seg)
    t2a = itk.DirectedHausdorffDistanceImageFilter.New(image_seg,image_gt)

    a2t.Update()
    t2a.Update()

    v1 = a2t.GetDirectedHausdorffDistance()
    v2 = t2a.GetDirectedHausdorffDistance()

    return max(v1, v2)

def precision(image, annotation):
    classes_labels = np.unique(annotation)

    score = np.zeros(len(classes_labels))

    score = np.zeros(len(classes_labels))

    score = precision_score(np.reshape(annotation, -1), np.reshape(image, -1), average = None)

    score = score[1:]

    return score

def recall(image, annotation):
    classes_labels = np.unique(annotation)

    score = np.zeros(len(classes_labels))

    score = recall_score(np.reshape(annotation, -1), np.reshape(image, -1), average=None)

    score = score[1:]

    return score