import numpy as np
import networkx as nx
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import distance_transform_edt

class Node():
    
    def __init__(self, pixels_array, value):
        self.pixels_array = pixels_array
        self.value = value

def create_graph_image(image_gray, predicted_mask):

    labelled_image = label(image_gray, connectivity=1)

    temp = __change_background_region(labelled_image)
    
    labelled_image = np.asarray(temp, np.uint8)

    labelled_image += 1

    regions = regionprops(labelled_image)

    nodes_data = __get_nodes_data(predicted_mask, regions)
    edges_data = __get_edges_data_TD(labelled_image, nodes_data)

    labels_name_image_test = []

    for i in range(0, len(nodes_data)):
        labels_name_image_test.append(str(i))

    Gi = __create_graphe(nodes_data, edges_data, labels_name_image_test)

    labelled_image -= 1

    return labelled_image, Gi, nodes_data, labels_name_image_test, regions

def create_graphe_model_from_csv(filepath, label, node_data):

    adjacence = np.genfromtxt(filepath, delimiter=',')
    x,y = adjacence.shape

    G = nx.Graph()

    for i in range(0, x):
        for j in range(0, y):
            if i == j:
                G.add_nodes_from([(label[i], dict(weight=node_data[i][j]))])

    for i in range(0, x):
        for j in range(0, y):
            if i != j:
                G.add_edges_from([(label[i], label[j], dict(weight=adjacence[i][j]))])

    return G

def create_model_labels(nb_classes):
    labels_m = []

    for i in range(0, nb_classes):
        labels_m.append(str(i))
    
    return labels_m

def create_model_nth_images(nb_classes):
    label_m = np.zeros((nb_classes, nb_classes, nb_classes))

    for i in range(0, nb_classes):
        for j in range(0, nb_classes):
            if i == j:
                label_m[i][j][i] = 1
    
    return label_m

def __create_graphe(nodes_data, edges_data, labels):

    G = nx.Graph()

    for i in range(0, len(nodes_data)):
        G.add_nodes_from([(labels[i],dict(weight=nodes_data[i].value))])

    for i in range(0, len(nodes_data)):
        for j in range(0, len(nodes_data)):
            G.add_edges_from([(labels[i],labels[j],dict(weight=edges_data[i][j]))])

    return G

def __get_nodes_data(image, region_properties):

    nodes = []

    for region in region_properties:

        if region.area >= 0:
            array_pixels = region.coords

            value = None
            for pixel in array_pixels:
                x = pixel[0]
                y = pixel[1]
                if value is None:
                    value = image[x][y]
                else:
                    value += image[x][y]

            value = value / len(array_pixels)
            
            nodes.append(Node(array_pixels, value))
    
    return nodes

def __get_edges_data_TD(images_regions, nodes_data):

    x,y = images_regions.shape

    temp = np.zeros((x, y), np.uint8)

    for i in range(0, len(nodes_data)):
        for coords in nodes_data[i].pixels_array:
            x = coords[0]
            y = coords[1]

            temp[x][y] = i
    
    temp += 1

    regions = regionprops(temp)

    Am = np.zeros((len(regions), len(regions)))

    for i in range(1, len(regions) + 1):

        mask = np.where(temp == i, 0, 1)
        dist = distance_transform_edt(mask)

        for j in range(1, len(regions) + 1):
            
            if j != i and Am[i-1][j-1] == 0 and Am[j-1][i-1] == 0:

                second_mask = np.where(temp == j, 1, 0)
                res = second_mask * dist

                Am[i-1][j-1] = Am[j-1][i-1] = np.min(res[np.nonzero(res)])

    return Am

def __change_background_region(labelled_image):

    x, y = np.shape(labelled_image)

    bg = np.where(labelled_image == 0, 1, 0)

    mask_bg = np.zeros((x, y), np.uint8)

    for i in range(0, x):
        for j in range(0, y):
            if i == 0 or i == x-1 or j == 0 or j == y-1:
                mask_bg[i][j] = 1

    inter = np.bitwise_and(bg, mask_bg)

    labelled_inter = label(inter, connectivity=1)
    regions_inter = regionprops(labelled_inter)

    labelled_bg = label(bg, connectivity=1)
    regions_bg = regionprops(labelled_bg)
    
    new_image = np.zeros((x, y))

    index = 0
    for rg_inter in regions_inter:
        for rg_bg in regions_bg:
            if __inList(rg_inter.coords[0], rg_bg.coords) :
                for coords in rg_bg.coords:

                    i = coords[0]
                    j = coords[1]

                    new_image[i][j] = 1

    index = np.max(new_image) + 1
    for rg_bg in regions_bg:
        xx = rg_bg.coords[0][0]
        yy = rg_bg.coords[0][1]
        if new_image[xx][yy] != 1 :
            for coords in rg_bg.coords:
                i = coords[0]
                j = coords[1]

                new_image[i][j] = index
            index +=1
    
    new_image -= 1

    nb_background = np.max(new_image)
    labelled_image = np.asarray(labelled_image) + nb_background

    for i in range(0, x):
        for j in range(0, y):
            if new_image[i][j] != -1:
                labelled_image[i][j] = new_image[i][j]
    
    return labelled_image

def __inList(array, list):
    for element in list:
        if np.array_equal(element, array):
            return True
    return False