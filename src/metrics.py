import numpy as np
import math
from sklearn.metrics import precision_score, recall_score
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import itk
import SimpleITK as sitk
from skimage.measure import regionprops
import cc3d

from skimage.measure import regionprops, label

def keep_largest_component(mask):
    lbl = label(mask, connectivity=2)
    regions = regionprops(lbl)

    max_size = 0
    max_label = None
    for region in regions:
        if region.area > max_size:
            max_label = region.label
            max_size = region.area
    
    mask = np.where(lbl == max_label, 1, 0)

    return mask


def dice_score(seg, gt):

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels) + 1)

    s = 0

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        inter = np.logical_and(mask_gt,mask_seg)

        val = 2 * np.sum(inter) / (np.sum(mask_gt) + np.sum(mask_seg)) 

        score[i] = val
        s += val

    score[len(classes_labels)] = s / len(classes_labels)
    
    return score

def prec_score(seg, gt):
    
    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels) + 1)

    s = 0

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        val = precision_score(mask_gt, mask_seg)

        score[i] = val
        s += val

    score[len(classes_labels)] = s / len(classes_labels)
    
    return score

def rec_score(seg, gt):
    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels) + 1)

    s = 0

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        val = recall_score(mask_gt, mask_seg)

        score[i] = val
        s += val

    score[len(classes_labels)] = s / len(classes_labels)
    
    return score

def Hausdorff_score(seg, gt):

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        image_seg = itk.GetImageFromArray(mask_seg)
        image_gt = itk.GetImageFromArray(mask_gt)

        a2t = itk.DirectedHausdorffDistanceImageFilter.New(image_gt,image_seg)
        t2a = itk.DirectedHausdorffDistanceImageFilter.New(image_seg,image_gt)

        a2t.Update()
        t2a.Update()

        score[i] = max(directed_hausdorff(mask_gt, mask_seg)[0], directed_hausdorff(mask_seg, mask_gt)[0])
    
    return score

def all_informations_data_frame(datas, seg, gt):
    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = np.zeros((len(classes_labels),len(datas)))

    index = []

    for i in range(1, len(classes_labels)):
        index.append("C" + str(i))
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        inter = np.logical_and(mask_gt,mask_seg)

        gt_flatten = mask_gt.flatten()
        seg_flatten = mask_seg.flatten()

        pos = 0

        if "dice" in datas:
            score[i-1][pos] = 2 * np.sum(inter) / (np.sum(mask_gt) + np.sum(mask_seg))
            pos+=1

        if "precision" in datas:
            score[i-1][pos] = precision_score(gt_flatten, seg_flatten)
            pos +=1

        if "recall" in datas:
            score[i-1][pos] = recall_score(gt_flatten, seg_flatten)
            pos+=1
        
        if "hausdorff" in datas:
            image_seg = sitk.GetImageFromArray(mask_seg)
            image_gt = sitk.GetImageFromArray(mask_gt)

            haus = sitk.HausdorffDistanceImageFilter()
            haus.Execute(image_seg,image_gt)

            score[i-1][pos] = haus.GetHausdorffDistance()
            pos+=1

        if "nb_CC" in datas:
            score[i-1][pos] = len(regionprops(cc3d.connected_components(mask_seg)))
            pos+=1

        if "box_dice" in datas:
            mask_gt = keep_largest_component(mask_gt)
            gd_box = __bbox2(mask_gt)
            image_box = __bbox2(mask_seg)

            inter = np.logical_and(gd_box, image_box)
            union = np.logical_or(gd_box, image_box)

            val = np.sum(inter) / np.sum(union)

            score[i-1][pos] = val
            pos+=1

    pos = 0

    if "dice" in datas:
        score[len(classes_labels)-1][pos] = np.sum(score[0:len(classes_labels),pos]) / len(classes_labels)
        pos+=1
    
    if "precision" in datas:
        score[len(classes_labels)-1][pos] = np.sum(score[0:len(classes_labels),pos]) / len(classes_labels)
        pos+=1
    
    if "recall" in datas:
       score[len(classes_labels)-1][pos] = np.sum(score[0:len(classes_labels),pos]) / len(classes_labels)
       pos+=1
    
    if "hausdorff" in datas:
       score[len(classes_labels)-1][pos] = np.sum(score[0:len(classes_labels),pos]) / len(classes_labels)
       pos+=1

    if "nb_CC" in datas:
        score[len(classes_labels)-1][pos] = np.sum(score[0:len(classes_labels),pos]) / len(classes_labels)
        pos+=1

    index.append("Mean")

    df = pd.DataFrame(score, index = index, columns = datas)

    return df

def __bbox2(img):
    rows = np.any(img, axis=0)
    cols = np.any(img, axis=1)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    box = np.copy(img)

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            box[i][j] = 1

    lbl = label(box)
    regions = regionprops(lbl)

    for region in regions:
        xmin, ymin, xmax, ymax = region.bbox

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            box[i][j] = 1

    return box