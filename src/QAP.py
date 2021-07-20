from numpy.linalg import norm
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def apply_QAP(Kv, Ke, permutations, alpha):
    
    S = []
    
    for X in permutations:
        # We flatten X to obtain a vector
        vec_x = X.flatten('F')
        x_translate = np.transpose(vec_x)

        # Calculation of the dissimilarities score
        score_kv = np.dot(np.matmul(x_translate,Kv), vec_x)
        score_Ke = np.dot(np.matmul(x_translate,Ke), vec_x)

        S.append(alpha * score_kv + (1-alpha) * score_Ke)

    # Sorting the score
    S = np.asarray(S)
    inds = np.argsort(S)
    S = S[inds]

    Matches = []
    for i in inds:
        Matches.append(permutations[i])

    return S, Matches

def calculate_Ke(Am, Ar, lbd, Cs):
    """[summary]

    Args:
        Am ([type]): [description]
        Ar ([type]): [description]
        lbd ([type]): [description]
        Cs ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    xm, ym, zm = Am.shape
    xi, yi, zi = Ar.shape

    nb_columns_rows = ym * yi

    Ke = np.zeros((nb_columns_rows, nb_columns_rows))

    for i in  range(0, nb_columns_rows):
        for j in range(0,nb_columns_rows):
            # Case dans ma matrice K
            indice_case_row = math.floor(i / yi)
            indice_case_column = math.floor(j / yi)
            
            # Position dans la case
            indice_case_inside_row = i % yi
            indice_case_inside_column = j % yi

            # Obtention des vecteur des positions
            value_edge_model = Am[:, indice_case_row, indice_case_column]
            value_edge_test = Ar[:, indice_case_inside_row, indice_case_inside_column]

            # Evaluation des dissimilarités
            norm1 = norm(value_edge_model)
            norm2 = norm(value_edge_test)

            scalar_pro = np.dot(value_edge_model, value_edge_test)

            if norm1 !=0 and norm2 !=0:
                cos_angle = scalar_pro / ( norm1 * norm2)

                Ke[j][i] = lbd * math.fabs(cos_angle - 1)/2 + (1-lbd) * math.fabs(norm1 - norm2) / Cs

    return Ke

def calculate_Kv(nodes_model, nodes_unsure, Am, Ar):
    """[summary]

    Args:
        nodes_model ([type]): [description]
        nodes_unsure ([type]): [description]
        Am ([type]): [description]
        Ar ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    xm, ym = Am[0,:,:].shape
    xi, yi = Ar[0,:,:].shape

    nb_columns_rows = ym * yi

    Kv = np.zeros((nb_columns_rows, nb_columns_rows))

    for i in  range(0, nb_columns_rows):
        
        # position dans la matrice Kv
        indice_model = math.floor(i / xi)
        indice_test = i % xi

        # Evaluation des dissimilarités
        value_node_model = nodes_model[indice_model].probability_vector
        value_node_test = nodes_unsure[indice_test].probability_vector

        Kv[i][i] = mean_squared_error(value_node_model, value_node_test)

    return Kv

def define_permutations(regions, nodes_matching, nb_classes):
    """Algorithms to define the permutations that will be used with the QAP on a one-to-one matching.

    Args:
        regions (list): the regions obtain from scipy regionprops applied to the image.
        nb_classes (int): the number of classes of the system.

    Returns:
        list: A list of numpy array describing all the permutations possibilities.
    """

    all_permutations = []
    
    initial_regions_labels = np.zeros((len(nodes_matching), nb_classes))

    indice = 0
    for node in nodes_matching:
        region = regions[node.ids-1]
        label = region.max_intensity
        max_id = label - 1
        initial_regions_labels[indice][max_id] = 1
        indice += 1

    dico = {}

    for i in range(0, nb_classes):
        if np.sum(initial_regions_labels[:,i]) > 1:
            indices = np.squeeze(np.where(initial_regions_labels[:,i] != 0), axis=0)
            temp = []
            for indice in indices:
                t =  np.zeros(len(nodes_matching))
                t[indice] = 1
                temp.append(t)
            dico[i] = temp

    all_permutations.append(initial_regions_labels)

    for k, v in dico.items():

        temp = []
        while(len(all_permutations) != 0):
            val = all_permutations.pop()

            for row in v:
                val[:,k] = row
                temp.append(np.copy(val))
        all_permutations = temp
        
    return all_permutations