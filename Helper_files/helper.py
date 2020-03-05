import numpy as np
import math
import networkx as nx
import itertools
import imageio
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from skimage.measure import label, regionprops
import cv2
from enum import Enum
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, distance_transform_bf


def create_test_data():
    
    im = cv2.imread('test_2.png')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    temp = np.zeros((13,13,6))

    graymap = {
        0:0,
        255:1,
        76:2,
        150:3,
        30:4,
        212:5
    }

    x,y = im.shape

    for i in range(0,x):
        for j in range(0,y):
            value = graymap[im[i][j]]
            temp[i][j][value] = 1

    return temp

def plot_image_nth_dims(rgb, image):
    
    x,y,z = image.shape

    step = math.floor(255 / z)

    temp = np.zeros((x,y))

    for k in range(0,z):

        canal = image[:,:,k]
        value = k * step

        for i in range(0,x):
            for j in range(0,y):
                if canal[i][j] == 1:
                    temp [i][j] = value
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(rgb)
    plt.subplot(1,2,2)
    plt.title('Segmentation')
    plt.imshow(temp)
    plt.show()

def get_region_properties(image):

    label_img = label(image, connectivity=1)
    
    regions = regionprops(label_img)

    return regions

def get_edges_data(nodes_data):

    adjacence = np.zeros((len(nodes_data), len(nodes_data)))

    for i in range(0, len(nodes_data)):
        for j in range(0,len(nodes_data)):

            if i != j:
                adjacence[i][j] = edge_metrics(nodes_data[i], nodes_data[j])
    
    return adjacence


def plot_graphe(G):
    
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos,
                        node_color='r',
                        node_size=500,
                        alpha=0.8)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=16)

    plt.axis('off')
    plt.show()

def plot_graphe_image(image_label, G):

    plt.figure('Graphe')
    pos = nx.spring_layout(G)  # positions for all nodes

    plt.subplot(1,2,1)
    # nodes
    nx.draw_networkx_nodes(G, pos,
                        node_color='r',
                        node_size=500,
                        alpha=0.8)
    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=16)

    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(image_label)
    plt.show()

def get_pixels_matching(regions_model, region_test, matching : dict):

    pixel_model = []
    pixel_test = []

    for k,v in matching.items():
        
        # model
        index = np.all(regions_model[:,:,k] == 1)
        m_pix = regions_model[index].coords[math.floor(len(regions_model[index].coords) / 2)]
        pixel_model.append(m_pix)

        # test
        index = int(v)
        t_pix = region_test[index].coords[math.floor(len(region_test[index].coords) / 3)]
        pixel_test.append(t_pix)

    return (pixel_model, pixel_test)

def plot_graph_matching(matching : dict,image_test):

    patches = []

    for k,v in matching.items():
        temp_patch = mpatches.Patch(color=(0,0,0), label = "test")
        patches.append(temp_patch)

    plt.imshow(image_test)

    m = plt.get_cmap()

    plt.legend(handles=patches)
    plt.show()
    

def calibrate_gray_scale_image(image):

    amin = np.amin(image)
    amax = np.amax(image)

    step = math.floor(255 / (amax - amin))

    x,y = image.shape
    temp = np.zeros((x, y))

    for i in range(0, x):
        for j in range(0, y):
            value = image[i][j]
            temp[i][j] = value * step
    
    return temp

def get_possibles_permutations(test, rows, cols, actual_column):
    results = []

    for i in range(actual_column, cols):
        
        index_meeting = []
        new_arrays = []

        for j in range(0, rows):
            if test[j][i] == 1:
                index_meeting.append(j)
        
        # Si la colonne est mauvaise (plusieurs 1 rencontrés)
        if len(index_meeting) > 1:

            for iteration in index_meeting:
                temp = np.copy(test)
                for index in index_meeting:
                    if iteration != index:
                        temp[index][i] = 0
                
                new_arrays.append(temp)

            for array_to_test in new_arrays:
                state = get_permutation_state(array_to_test, rows, cols)
                if state == Permutation.Unclear:
                    res = get_possibles_permutations(array_to_test, rows, cols, actual_column + 1)
                    if res != None and len(res) > 0:
                        for val in res:
                            results.append(val)
                elif state == Permutation.Valid:
                    results.append(array_to_test)
                
                elif state == Permutation.Impossible:
                    pass

            return results

    state = get_permutation_state(test, rows, cols)
    if state == Permutation.Valid:
        results.append(test)
    else:
        return None

    return results
            
def get_cols_permutations(array_test, nb_classes):
    
    # Obtention des vecteurs possibles par colonnes
    classes_cols = {}

    for i in range(0, nb_classes):

        vector = array_test[:,i]
        
        index_equal_one = []

        for j in range(0, len(vector)):
            if vector[j] == 1:
                index_equal_one.append(j)
        
        possibilities = []

        for iteration in index_equal_one:
            temp = np.copy(vector)
            for value in index_equal_one:
                if iteration != value:
                    temp[value] = 0
            possibilities.append(temp)
        
        classes_cols[i] = possibilities
    
    # Intialisation
    results = []

    for vector in classes_cols[0]:
        results.append(vector)

    for i in range(1, nb_classes):
        
        temp_results = []

        if i ==1:
            for val in results:
                for vector_to_add in classes_cols[i]:
                    actual_tab = np.asarray(val, np.int32)
                    new_col = np.asarray(vector_to_add, np.int32)

                    is_possible = True
                    
                    vector_to_test = actual_tab
                    inter = np.bitwise_and(vector_to_test, new_col)
                    if np.sum(inter) != 0:
                        is_possible = False
            
                    if is_possible:
                        new_array = np.column_stack((actual_tab, new_col))
                        temp_results.append(new_array)
            
            results = np.copy(temp_results)
            
        else:
            for val in results:
                for vector_to_add in classes_cols[i]:

                    actual_tab = np.asarray(val, np.int32)
                    new_col = np.asarray(vector_to_add, np.int32)

                    is_possible = True

                    x, y = actual_tab.shape

                    for k in range(0, y):
                        vector_to_test = actual_tab[:, k]
                        inter = np.bitwise_and(vector_to_test, new_col)
                        if np.sum(inter) != 0:
                            is_possible = False
                
                    if is_possible:
                        new_array = np.column_stack((actual_tab, new_col))
                        temp_results.append(new_array)
            
            results = np.copy(temp_results)

    return results

def check_if_permutation_already_exists(results, perm_to_add):
    
    flag_already_exist = False

    for res in results:
        if np.allclose(res, perm_to_add):
            flag_already_exist = True
    
    return flag_already_exist

def get_permutation_state(test, x, y):
    
    value = np.sum(test)

    flag_one_match = True

    for row in range(0, x):
        val = np.sum(test[row])
        if val > 1:
            flag_one_match = False
    
    for col in range(0, y):
        val = np.sum(test.T[col])
        if val > 1:
            flag_one_match = False

    if value == min(x, y) and flag_one_match == True:
        return Permutation.Valid
    elif value == min(x, y) and flag_one_match == False: 
        return Permutation.Impossible
    else:
        return Permutation.Unclear


def get_permutations_possibilities(nodes_data, classes):
    
    initial_regions_labels = np.zeros((len(nodes_data), len(classes)))

    for i in range(0, len(nodes_data)):

        if len(nodes_data[i].pixels_array) >= 20:
            index = 0
            value = 0
            for j in range(0, len(classes)):
                if nodes_data[i].value[j] > value:
                    value = nodes_data[i].value[j]
                    index = j
            
            initial_regions_labels[i][index] = 1

    initial_regions_labels = initial_regions_labels

    #temp = get_possibles_permutations(initial_regions_labels, len(nodes_data), len(classes), 0)
    temp = get_cols_permutations(initial_regions_labels, len(classes))

    M = []

    for permu in temp:
        M.append(np.asarray(permu).transpose())



    return M


def plot_matching(match_model, match_image, regions, image_test):
    
    image_plot = cv2.imread("Resources/face_canevas.png")

    image_plot = cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB)

    gmap = {
    (255,0,0) : 0,
    (127,0,0) : 1,
    (255,255,0) : 2,
    (255,0,255) : 3,
    (127,0,127) : 4,
    (0,0,255) : 5,
    (0,0,127) : 6,
    (0,255,255) : 7,
    (0,255,0) : 8
    }

    x,y,z = image_plot.shape

    temp = np.zeros((x, y))
    plot_dict = {}

    for i in range(0, x):
        for j in range(0, y):
            value = gmap[tuple(image_plot[i][j])]
            temp[i][j] = value

            if value in plot_dict.keys():
                pass
            else:
                plot_dict[value] = (i, j)
    
    pixel_model = []
    pixel_test = []

    for i in range(0, len(match_image)):
        
        # model
        index = match_model[i]
        m_pix = np.asarray(plot_dict[index])
        pixel_model.append(m_pix)

        # test
        index = match_image[i]
        t_pix = regions[index].coords[math.floor(len(regions[index].coords) / 3)]
        pixel_test.append(t_pix)
    
    new_image = np.concatenate((image_test, temp), axis=1)

    new_image = calibrate_gray_scale_image(new_image)

    res = cv2.cvtColor(np.asarray(new_image, dtype=np.uint8), cv2.COLOR_GRAY2RGB)

    for i in range(0, len(match_image)):
        t_pix = pixel_test[i]
        m_pix = pixel_model[i]
        m_pix[1] += image_test.shape[1]

        plot_value = 255 * (i / len(match_image))
        cv2.line(new_image, (t_pix[1], t_pix[0]), (m_pix[1], m_pix[0]), (plot_value, 255 - plot_value,  127), 4, 4)

    plt.imshow(new_image)
    #plt.show()





           

