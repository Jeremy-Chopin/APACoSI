import time
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.ndimage.morphology import distance_transform_edt
import pandas as pd
from scipy.ndimage import binary_fill_holes
import edt
from sklearn.metrics import mean_squared_error
import copy
from src.Nodes_data import Nodes_data

def export_matching(export_path, best_matching, regions, regions_label, model_image, test_image, test_matching, dims, intermediar_show = False):
    
    if dims == 2:
        export_matching_2d(export_path, best_matching, regions, regions_label, model_image, test_image, test_matching, intermediar_show)
    else:
        export_matching_3d(export_path, best_matching, regions, regions_label, model_image, test_image, test_matching, intermediar_show)
        

def export_matching_2d(export_path, best_matching, regions, regions_label, model_image, test_image, test_matching, intermediar_show):
    x,y = np.where(best_matching == 1)

    shape_x, shape_y = test_image.shape

    qap_matching = np.zeros((shape_x, shape_y))

    for p in range(0, len(x)):

        match_x, match_y = np.where(regions_label == x[p] + 1)

        qap_matching[match_x[0]][match_y[0]] = 255

    plt.subplot(2,2,1); plt.title("Node Model"); plt.imshow(model_image)
    plt.subplot(2,2,2); plt.title("Node Test"); plt.imshow(test_image)

    for region in regions:
        c_y, c_x = region.centroid
        plt.text(c_x, c_y, str(region.max_intensity), fontsize=16)

    plt.subplot(2,2,3); plt.title("QAP matching "); plt.imshow(qap_matching)

    c_x, c_y = np.where(Matches[0] == 1)

    for v in range(0, len(c_x)):
        c_yy, c_xx = regions[c_x[v]].centroid
        if Matches[0][c_x[v]][c_y[v]] == correct_matching[c_x[v]][c_y[v]]:
            plt.text(c_xx, c_yy, str(regions[c_x[v]].max_intensity), fontsize=16)
        else:
            plt.text(c_xx, c_yy, "X", fontsize=16)


    plt.subplot(2,2,4); plt.title("Correct matching"); plt.imshow(test_matching)

    for k, v in c_model.items():
        plt.text(v[1], v[0], str(k), fontsize=16)

    plt.savefig(export_path)

    if intermediar_show:
        plt.show()

    plt.close()

def export_matching_3d(export_path, best_matching, regions, regions_label, model_image, test_image, test_matching, intermediar_show):
    x,y = np.where(best_matching == 1)

    shape_x, shape_y, shape_z = test_image.shape

    qap_matching = np.zeros((shape_x, shape_y, shape_z))

    for p in range(0, len(x)):

        match_x, match_y, match_z = np.where(regions_label == x[p] + 1)

        qap_matching[match_x[0]][match_y[0]][match_z[0]] = 255


    x1,y1,z1 = np.where(model_image == 255)
    x2,y2,z2 = np.where(test_image == 255)
    x3,y3,z3 = np.where(qap_matching == 255)
    x4,y4,z4 = np.where(test_matching == 255)

    fig = plt.figure()

    axes = fig.add_subplot(221, projection='3d')
    axes.scatter(x1, y1, z1)
    axes.set_xlabel("Node Model")

    axes = fig.add_subplot(222, projection='3d')
    axes.scatter(x2, y2, z2)
    axes.set_xlabel("Node Test")

    axes = fig.add_subplot(223, projection='3d')
    axes.scatter(x3, y3, z3)
    axes.set_xlabel("Qap matching")

    axes = fig.add_subplot(224, projection='3d')
    axes.scatter(x4, y4, z4)
    axes.set_xlabel("Correct matching")

    plt.savefig(export_path)

    if intermediar_show:
        plt.show()

    plt.close()

def calculate_max_diagonal(image):
    
    axes = list(image.shape)

    somme = 0
    for axe in axes:
        somme += math.pow(axe, 2)
    
    return math.sqrt(somme)

def get_matching_elements_nodes_3d(matching, nodes_data_unsure):

    nodes_sure = []
    nodes_unsure = []

    indices = np.where(matching == 1)

    nodes_index = indices[0]
    classe = indices[1]

    for i in range(0,len(classe)):
        actual_classe = classe[i]
        node = nodes_data_unsure[nodes_index[i]]

        node.max_label_id = actual_classe + 1
        nodes_sure.append(node)

    for j in range(0, len(nodes_data_unsure)):
        if j not in nodes_index:
            nodes_unsure.append(nodes_data_unsure[j])

    return nodes_sure, nodes_unsure

def create_image_from_nodes(nodes_data, image_node):

    x,y,z = image_node.shape

    temp = np.zeros((x,y,z), dtype=np.uint8)

    for data in nodes_data:
        for node in data:
            temp = np.where(image_node == node.ids, node.max_label_id, temp)
    
    return temp

def signed_transform(mask):

    mask = mask.astype(np.bool)

    negatif = np.where(mask == True, False, True)
    
    exterior_value = edt.edt(negatif)

    mask_filled = binary_fill_holes(mask)

    negatif_filled = np.where(mask_filled == True, False, True)

    interior_value = edt.edt(negatif_filled)

    complete_mask = np.where(interior_value != exterior_value, -exterior_value, exterior_value)

    return complete_mask

def get_max_EDT(image_labelled, regions, label1, label2):
    for ids1 in label1:
        max_value = -np.inf
        mask1 = np.where(image_labelled == regions[ids1-1].label, False, True)
        dist = edt.edt(mask1, order='C', parallel=0)
        for ids2 in label2:
            mask2 = np.where(image_labelled == regions[ids2-1].label, 1, 0)
            res = dist * mask2

            max_value = max(np.max(res[np.nonzero(res)]), max_value)

    return max_value

def get_max_EDT_signed(image_labelled, regions, label1, label2):
    for ids1 in label1:
        max_value = -np.inf
        mask1 = np.where(image_labelled == regions[ids1-1].label, 1, 0)
        dist = signed_transform(mask1)
        for ids2 in label2:
            mask2 = np.where(image_labelled == regions[ids2-1].label, 1, 0)
            res = dist * mask2

            max_value = max(np.max(res[np.nonzero(res)]), max_value)

    return max_value

def update_An(An_initial, matching, pr_mask, regions, image_labelled, label_to_test, nb_classes):
    
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

def construct_Kv(An):
        
    nb_classes = An.shape[0]
    Kv = np.zeros((nb_classes * nb_classes, nb_classes * nb_classes))

    for i in range(0, nb_classes * nb_classes):
        row = int(i / nb_classes)
        col = i % nb_classes
        Kv[i][i] = An[row][col]

    return Kv

def calculate_matching_cost(alpha, Kv, Ke, nb_classes):
    
    X = np.zeros((nb_classes, nb_classes))
    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i == j:
                X[i][j] = 1

    vec_x = X.flatten('F')

    x_translate = np.transpose(vec_x)

    tempo_ke = np.matmul(x_translate,Ke)
    score_Ke = np.dot(tempo_ke,vec_x)

    tempo_kv = np.matmul(x_translate,Kv)
    score_Kv = np.dot(tempo_kv,vec_x)

    return (1-alpha)* score_Ke + alpha * score_Kv

def create_nodes_model(nb_classes):
    nodes_model = []

    for i in range(0,nb_classes):

        vector = np.zeros((nb_classes))
        vector[i] = 1

        node = Nodes_data(
            ids = i,
            max_label_id = i,
            probability_vector = vector,
            is_confusion=False,
            is_classified=False
        )

        nodes_model.append(node)
    return nodes_model