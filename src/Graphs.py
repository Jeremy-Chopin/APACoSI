from src.Nodes_data import Nodes_data
from skimage.measure import regionprops, label
from sklearn.metrics import mean_squared_error
import numpy as np
import cc3d
import math
import pandas as pd
import copy

def load_structural_model(filepath, nb_classes):
    """Function to load the structural knowledges of the application.

    Args:
        filepath (string): Path to the file containing the data.
        nb_classes (int): Number of classes considered in the application.

    Returns:
        Adjacency matrix (numpy array): Adjacency matrix of the system.
    """

    if ".npy" in filepath: 
        Am = np.load(filepath)
    elif ".csv" in filepath:
        Am = np.genfromtxt(filepath, delimiter=';')
    else:
        raise Exception("Structural knowledges file not in a supported type.")
    
    """if len(Am.shape) > 2:
        Am = np.transpose(Am, (2,0,1))"""

    return  Am

def define_nodes(image, predicted_mask, limit_matching_nodes = 2, limit_refinement_nodes = 4):
    """Defines matching nodes and refinement nodes to create from the considered image. Matching nodes are used for the initial matching and the others for the refinement step.

    Args:
        image (numpy array): image being processed
        predicted_mask (numpy array): Segmentation map of theprocessed image with proabilities 
        limit_matching_nodes (int, optional): Nodes created for each class that are used for the initial matching. Defaults to 2.
        limit_refinement_nodes (int, optional): Nodes created for each class that are used for the refinement. Defaults to 4.

    Returns:
        [type]: [description]
    """

    lbl_image = label(image, connectivity=2)

    regions = regionprops(lbl_image, image)

    print("There is " + str(len(regions)) + " connected components !")

    regions_ids, regions_counts = sort_connected_components_on_size(lbl_image, regions)

    nodes_matching = []
    nodes_refinement = []

    for i in regions_counts.keys():

        ids = np.asarray(regions_ids[i])
        size = np.asarray(regions_counts[i])

        inds = np.argsort(-1 * size)

        sortedids = ids[inds]
        sortedsize = size[inds]

        for k in range(0, len(sortedids)):
            if k < limit_refinement_nodes:

                region = regions[sortedids[k] - 1]
                region_mask = np.where(lbl_image == region.label, 1, 0)

                if len(lbl_image.shape) > 2:
                    proba_mask = np.expand_dims(region_mask, axis=3) * predicted_mask

                    region_probs = np.sum(proba_mask, axis=(0,1,2)) / np.sum(region_mask)
                else:
                    proba_mask = np.expand_dims(region_mask, axis=2) * predicted_mask

                    region_probs = np.sum(proba_mask, axis=(0,1)) / np.sum(region_mask)

                max_label = np.argmax(region_probs)

                node = Nodes_data(
                            ids= (region.label),
                            max_label_id= max_label + 1,
                            probability_vector= region_probs,
                            is_confusion= True,
                            is_classified = False)

                if k < limit_matching_nodes:
                    nodes_matching.append(node)
                elif k < limit_refinement_nodes:
                    nodes_refinement.append(node)
    
    list_ids = []

    # Réorganisation des noeuds pour qu'ils soient en accord avec les régions
    nodes = []
    for region in regions:
        for node in nodes_matching:
            if node.ids == region.label:
                nodes.append(node)
                list_ids.append(node.ids)
                break

    nodes_matching = nodes

    return lbl_image, regions, nodes_matching, nodes_refinement

def sort_connected_components_on_size(lbl_image, regions):

    region_counts = {}
    region_ids = {}

    for region in regions:
        if region.max_intensity in region_counts.keys():
            region_counts[region.max_intensity].append(region.area)
            region_ids[region.max_intensity].append(region.label)
        else:
            region_counts[region.max_intensity] = [region.area]
            region_ids[region.max_intensity] = [region.label]
    
    return region_ids, region_counts

def filter_image_from_selected_nodes(labelled_image, nodes_matching, nodes_refinement):

    x,y,z = labelled_image.shape

    temp = np.zeros((x,y,z), dtype=np.uint8)

    for node in nodes_matching:
        temp = np.where(labelled_image == node.ids, node.max_label_id, temp)

    for node in nodes_refinement:
        temp = np.where(labelled_image == node.ids, node.max_label_id, temp)
    
    return temp         

def calculate_matching_cost(X, Kv, Ke):

    vec_x = X.flatten('F')
    x_translate = np.transpose(vec_x)

    # Calculation of the dissimilarities score
    score_kv = np.dot(np.matmul(x_translate,Kv), vec_x)
    score_Ke = np.dot(np.matmul(x_translate,Ke), vec_x)

    return score_kv + score_Ke