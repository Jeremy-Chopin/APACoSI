import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def non_matching_regions(regions, nodes_data, image_labels, match_test):
    
    x,y = image_labels.shape
    temp = np.zeros((x, y))
    
    for i in range(0, len(regions)):

        proba = sorted(nodes_data[i].value, reverse= True)

        if i in match_test:
            index = list(nodes_data[i].value).index(proba[0])
        else:
            index = list(nodes_data[i].value).index(proba[1])

        for coords in regions[i].coords:
            x = coords[0]
            y = coords[1]

            temp[x][y] = index + 1

    return temp

def non_matching_regions_refinment(regions, image_labels, final_matching):
    
    x,y = final_matching.shape
    temp = np.copy(image_labels)

    for i in range(0, x):
        for j in range(0, y):
            if final_matching[i][j] == 1:
                for coords in regions[j].coords:
                    xx = coords[0]
                    yy = coords[1]

                    temp[xx][yy] = i + 1
    
    return temp

def get_iou(groundtruth, image_corrected, nb_classes, classes):

    x, y = groundtruth.shape

    gd_masks = np.zeros((x, y, nb_classes + 1), np.uint8)

    image_masks = np.zeros((x, y, nb_classes + 1), np.uint8)

    for i in range(0, x):
        for j in range(0, y):

            gd_value = int(groundtruth[i][j])
            gd_masks[i][j][gd_value] = 1

            image_value = np.asarray(image_corrected[i][j], dtype= np.uint8)
            image_masks[i][j][image_value] = 1

    for i in range(1, nb_classes):
        lbl = label(gd_masks[:,:,i])
        regions = regionprops(lbl)

        lbl_value = 0
        max_area = 0
        for region in regions:
            if region.area > max_area:
                lbl_value = region.label
                max_area = region.area
        
        

        gd_masks[:,:,i] = np.where(lbl == lbl_value, 1, 0)

    # IoU
    res = []
    for i in range(0, nb_classes):
        
            inter = cv2.bitwise_and(gd_masks[:,:,i], image_masks[:,:,i+1])
            union = cv2.bitwise_or(gd_masks[:,:,i], image_masks[:,:,i+1])
            
            res.append(round(100 * (np.sum(inter) / np.sum(union)), 1))
    
    # Box IoU
    res_box = []
    for i in range(0, nb_classes):

            gd_box = __bbox2(gd_masks[:,:,i])
            image_box = __bbox2(image_masks[:,:,i+1])
        
            inter = cv2.bitwise_and(gd_box, image_box)
            union = cv2.bitwise_or(gd_box, image_box)

            res_box.append(round(100 * (np.sum(inter) / np.sum(union)), 1))

    return res, res_box


def create_csv(data1 : list, data2 : list, filepath, classes):
    data_1 = np.asarray(data1)
    data_2 = np.asarray(data2)

    res = np.stack((data1, data2))

    df = pd.DataFrame(res, index = ['sans','avec'], columns = classes)

    cs = df.to_csv(filepath)

def __bbox2(img):
    rows = np.any(img, axis=0)
    cols = np.any(img, axis=1)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    box = np.copy(img, np.uint8)

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