from matplotlib.pyplot import axis
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

    score = np.zeros(len(classes_labels))

    s = 0

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0)
        mask_seg = np.where(seg == i, 1, 0)

        inter = np.logical_and(mask_gt,mask_seg)

        val = 2 * np.sum(inter) / (np.sum(mask_gt) + np.sum(mask_seg)) 

        score[i] = val
        s += val

    score = score[1:]

    mean_score = np.mean(score, axis=0, keepdims=True)

    score = np.concatenate((score, mean_score), axis=0)
    
    return score

def prec_score(seg, gt):
    
    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    score = precision_score(np.reshape(gt, -1), np.reshape(seg, -1), average = None)

    score = score[1:]

    mean_score = np.mean(score, axis=0, keepdims=True)

    score = np.concatenate((score, mean_score), axis=0)

    return score

def rec_score(seg, gt):
    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    classes_labels = np.unique(gt)

    score = recall_score(np.reshape(gt, -1), np.reshape(seg, -1), average=None)

    score = score[1:]

    mean_score = np.mean(score, axis=0, keepdims=True)

    score = np.concatenate((score, mean_score), axis=0)
    
    return score

def Hausdorff_score(seg, gt):

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    for i in range(0, len(classes_labels)):
        mask_gt = np.where(gt == i, 1, 0).astype(np.float32)
        mask_seg = np.where(seg == i, 1, 0).astype(np.float32)

        image_seg = itk.GetImageFromArray(mask_seg)
        image_gt = itk.GetImageFromArray(mask_gt)

        a2t = itk.DirectedHausdorffDistanceImageFilter.New(image_gt,image_seg)
        t2a = itk.DirectedHausdorffDistanceImageFilter.New(image_seg,image_gt)

        a2t.Update()
        t2a.Update()

        score[i] = max(a2t.GetDirectedHausdorffDistance(), t2a.GetDirectedHausdorffDistance())
    
    score = score[1:]

    mean_score = np.mean(score, axis=0, keepdims=True)

    score = np.concatenate((score, mean_score), axis=0)
    
    return score

def nb_cc_score(seg, gt):

    classes_labels = np.unique(gt)

    score = np.zeros(len(classes_labels))

    for i in range(0, len(classes_labels)):
        mask_seg = np.where(seg == i, 1, 0)

        score[i] = len(regionprops(cc3d.connected_components(mask_seg)))
    
    score = score[1:]

    mean_score = np.mean(score, axis=0, keepdims=True)

    score = np.concatenate((score, mean_score), axis=0)

    return score

def all_informations_data_frame(datas, seg, gt):
    classes_labels = np.unique(gt)

    scores = []

    for d in datas:
        if d == 'dice':
            scores.append(dice_score(seg, gt))

        if d == 'precision':
            scores.append(prec_score(seg, gt))

        if d == 'recall':
            scores.append(rec_score(seg, gt))

        if d == 'hausdorff':
            scores.append(Hausdorff_score(seg, gt))

        if d == 'nb_CC':
            scores.append(nb_cc_score(seg, gt))
    
    index = []
    for i in range(1, len(classes_labels)):
        index.append('C{}'.format(i))

    index.append('Avg')

    arr = np.stack(scores, axis=0)

    arr = np.transpose(arr, (1,0))

    df = pd.DataFrame(arr, index=index, columns=datas)

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