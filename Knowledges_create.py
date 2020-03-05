import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.measure import regionprops, label
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, distance_transform_bf

"""
Distance
"""

def distances_knowledge(RGB_dir, SEG_dir, nb_classes, save_file_path, delimiter=","):
    list_images = os.listdir(RGB_dir)

    adjacence = np.zeros((nb_classes, nb_classes))

    for image_name in list_images:

        image_path = os.path.join(y_test_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        temp = define_elements_distances(image)

        for i in range(0, nb_classes):
            for j in range(0, nb_classes):
                if i != j:
                    adjacence[i][j] += temp[i][j]

    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i != j:
                    adjacence[i][j] = adjacence[i][j] / len(list_images)

    np.savetxt(save_file_path, adjacence, delimiter=delimiter)

def define_elements_distances(image):
    
    x, y = image.shape
    
    # Nombre de pixel appartennant à une classe
    size = {}

    # Somme des coordonées de pixels (x, y) de chaque classe
    x_dict = {}
    y_dict = {}

    # On navigue sur l'ensemble des pixels de l'image, et on renseigne les informations de tailles et de coordonées
    for i in range(0, x):
        for j in range(0, y):
            value = image[i][j]
            if value in size:
                size[value] += 1
                x_dict[value] += i
                y_dict[value] += j
            else:
                size[value] = 1
                x_dict[value] = i
                y_dict[value] = j

    # Pour chaque classe, on obtient le barycentre moyen
    for key in size.keys():
        nb_elements = size[key]
        x_dict[key] = x_dict[key] / nb_elements
        y_dict[key] = y_dict[key] / nb_elements

    nb_classes = len(size.keys())
    adjacence = np.zeros((nb_classes, nb_classes))

    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i != j:
                adjacence[i][j] = get_distances_face_elements_metrics(x_dict[i], x_dict[j], y_dict[i], y_dict[j])

    return adjacence

def get_distances_face_elements_metrics(x1, x2, y1, y2):
    val = math.pow(x1-x2, 2) + math.pow(y1-y2, 2)
    distance = math.sqrt(val)
    return distance

"""
Transform distance
"""

def transform_distances_knowledge(RGB_dir, SEG_dir, nb_classes, save_file_path, delimiter=","):
    list_images = os.listdir(RGB_dir)

    adjacence = np.zeros((nb_classes, nb_classes))

    for image_name in list_images:

        image_path = os.path.join(y_test_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        labelled = label(image, neighbors=8)

        temp = define_elements_transform_distances(image, labelled, nb_classes)

        for i in range(0, nb_classes):
            for j in range(0, nb_classes):
                if i != j:
                    adjacence[i][j] += temp[i][j]

    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i != j:
                    adjacence[i][j] = adjacence[i][j] / len(list_images)

    np.savetxt(save_file_path, adjacence, delimiter=delimiter)

def define_elements_transform_distances(image, labelled, nb_classes):

    regions = regionprops(labelled, image)

    image_2 = erase_artefacts_region(image, regions)

    Am = np.zeros((nb_classes, nb_classes))

    for i in range(1, nb_classes + 1):

        mask = np.where(image_2 == i, 0, 1)
        dist = distance_transform_edt(mask)

        for j in range(1, nb_classes + 1):
            
            if j != i and Am[i-1][j-1] == 0 and Am[j-1][i-1] == 0:

                second_mask = np.where(image_2 == j, 1, 0)
                res = second_mask * dist

                Am[i-1][j-1] = Am[j-1][i-1] = np.min(res[np.nonzero(res)])

    return Am

def erase_artefacts_region(image_label, regions):
    
    x,y = image_label.shape

    mapping = {}

    for region in regions:
        #coord = region.coords[0]
        #value = image_label[coord[0], coord[1]]
        
        value = region.min_intensity

        if value in mapping.keys():
            if mapping[value].area < region.area:
                mapping[value] = region
        else:
            mapping[value] = region
            

    temp = np.zeros((x, y), np.uint8)

    for i in range(0, x):
        for j in range(0, y):
            if image_label[i][j] == 0:
                temp[i][j] = 1

    for key in mapping.keys():
        for coords in mapping[key].coords:
            x = coords[0]
            y = coords[1]

            temp[x][y] = key + 1

    return temp


DATA_DIR = './Datasets/data_face_simple'
nb_classes = 9

x_test_dir = os.path.join(DATA_DIR, 'RGB')
y_test_dir = os.path.join(DATA_DIR, 'Labels')

distances_knowledge(x_test_dir, y_test_dir, nb_classes, "face_elements_distances.csv", ",")

transform_distances_knowledge(x_test_dir, y_test_dir, nb_classes, "face_elements_transform_distances.csv", ",")