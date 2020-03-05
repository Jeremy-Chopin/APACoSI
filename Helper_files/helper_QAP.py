from enum import Enum
import numpy as np
import networkx as nx
import math


def apply_QAP(Gi, Gm, labels_Gi, labels_Gm, nodes_data,  classes, alpha = None):

    Ai=nx.adjacency_matrix(Gi,nodelist=labels_Gi).toarray()
    Am=nx.adjacency_matrix(Gm,nodelist=labels_Gm).toarray()

    M = __get_permutations_possibilities(nodes_data, classes)

    if alpha == None:
        K = __get_matrice_K(Am,Ai,Gm, Gi, labels_Gm, labels_Gi)
    else:
        K = __get_matrice_K(Am,Ai,Gm, Gi, labels_Gm, labels_Gi, alpha)

    best_score, best_matching,  worst_score, worst_matching = __find_best_worst_matching(K, M)

    return best_score, best_matching

def apply_QAP_with_refinment(Gi, Gm, labels_Gi, labels_Gm, nodes_data,  classes, alpha = None):

    Ai=nx.adjacency_matrix(Gi,nodelist=labels_Gi).toarray()
    Am=nx.adjacency_matrix(Gm,nodelist=labels_Gm).toarray()

    M = __get_permutations_possibilities(nodes_data, classes)

    if alpha == None:
        K = __get_matrice_K(Am,Ai,Gm, Gi, labels_Gm, labels_Gi)
    else:
        K = __get_matrice_K(Am,Ai,Gm, Gi, labels_Gm, labels_Gi, alpha)

    best_score, best_matching,  worst_score, worst_matching = __find_best_worst_matching(K, M)

    # Operation de corrections pour matcher les élements restants
    #__refinment_matching(K, best_matching)

    return best_score, best_matching, K

def refinment_matching(K, initial_matching, regions, match_test):

    
    x, y = initial_matching.shape

    new_matching = np.zeros((x,len(regions)), np.uint8)

    for i in range(0, len(match_test)):
        new_matching[i][int(match_test[i])] = 1

    for i in range(0, y):
        if np.sum(new_matching[:, i]) == 0:
            score = np.inf
            index = -1
            for j in range(0, x):

                new_matching[j][i] = 1

                new_score = __get_cost_matching(K, new_matching)

                if new_score < score:
                    score = new_score
                    index = j
                
                new_matching[j][i] = 0
    
            new_matching[index][i] = 1
            
    return new_matching

def refinment_matching_second(K, initial_matching, regions, nodes_data, match_test, Ai, alpha):
    
    x, y = initial_matching.shape

    new_matching = np.zeros((x,len(regions)), np.uint8)

    for i in range(0, len(match_test)):
        new_matching[i][int(match_test[i])] = 1

    for i in range(0, y):
        if np.sum(new_matching[:, i]) == 0:

            proba = nodes_data[i].value

            dist_vect = np.zeros((1,9))
            probs = np.zeros((1,9))

            for j in range(0, 9):
                label_region_repr = int(match_test[j])
                dist_vect[0][j] = 1 / Ai[i][label_region_repr]
                
                probs[0][j] = proba[j] / np.max(proba)
            
            score = alpha * probs + (1 - alpha) * dist_vect
            print(score)

            ind = np.argmax(score)
            print(ind)

            new_matching[ind][i] = 1

            
    return new_matching

def refinment_matching_third(K, initial_matching, regions, nodes_data, match_test, Ai, alpha):
    
    x, y = initial_matching.shape

    new_matching = np.zeros((x,len(regions)), np.uint8)
    all_probs = []

    for i in range(0, len(match_test)):
        new_matching[i][int(match_test[i])] = 1

        all_probs.append(nodes_data[int(match_test[i])].value)


    for i in range(0, y):
        if np.sum(new_matching[:, i]) == 0:

            proba = nodes_data[i].value

            dist_vect = np.zeros((1,9))
            probs = np.zeros((1,9))

            for j in range(0, 9):
                label_region_repr = int(match_test[j])
                dist_vect[0][j] = Ai[i][label_region_repr]
                
                proba_id = all_probs[j]

                somme = 0
                for k in range(0,len(proba)):
                    somme += math.pow(proba[k] - proba_id[k],2)
                res = math.sqrt(somme)
                probs[0][j] = res
            
            probs = probs / np.max(probs)
            dist_vect = dist_vect / np.max(dist_vect)

            score = alpha * probs + (1 - alpha) * dist_vect

            ind = np.argmin(score)

            new_matching[ind][i] = 1

            
    return new_matching

def get_matching_elements_labels(matching, labels_model, labels_image):

    match_model = []
    match_image = []

    for i in range(0,len(labels_model)):
        for j in range(0,len(labels_image)):
            if matching[i][j] == 1:
                match_model.append(int(labels_model[i]))
                match_image.append(int(labels_image[j]))

    return match_model, match_image

def __get_matrice_K(A_model,A_test, G_model, G_test, labels_model, labels, alpha = None):
    x1, y1 = A_model.shape
    x2, y2 = A_test.shape

    nodes_model = nx.nodes(G_model)._nodes
    nodes_test = nx.nodes(G_test)._nodes

    nb_columns_rows = x1 * x2

    K = np.zeros((nb_columns_rows, nb_columns_rows))

    diag_max_val = n_diag_max_val = 0
    
    for i in  range(0, nb_columns_rows):
        for j in range(0,nb_columns_rows):
            if i == j :
                indice_model = math.floor(i / x2)
                indice_test = j % x2

                value_node_model = nodes_model[labels_model[indice_model]]
                value_node_test = nodes_test[labels[indice_test]]

                K[i][j] = __mesure_difference_nodes(value_node_model, value_node_test, "weight")
            else:
                # Case dans ma matrice K
                indice_case_row = math.floor(i / x2)
                indice_case_column = math.floor(j/x2)
                
                # Position dans la case
                indice_case_inside_row = i % x2
                indice_case_inside_column = j % x2

                # Valeur des edges
                value_edge_modele = A_model[indice_case_row][indice_case_column]
                value_edge_test = A_test[indice_case_inside_row][indice_case_inside_column]

                # Remplissage de la matrice K
                if(value_edge_modele != 0  and value_edge_test != 0):
                    K[i][j] = __mesure_difference_edges(value_edge_modele, value_edge_test)
                    if K[i][j] >n_diag_max_val:
                        n_diag_max_val = K[i][j]

    if alpha != None:
        for i in  range(0, nb_columns_rows):
            for j in range(0,nb_columns_rows):
                if i == j :
                    K[i][j] =  alpha * K[i][j]
                else:
                    K[i][j] =  (1 - alpha) * K[i][j] / n_diag_max_val

    return K

def __get_cost_matching(K, X_ref):

    x,y = X_ref.shape

    vec_x = np.reshape(X_ref, (x*y,1))
    x_translate = np.transpose(vec_x)

    tempo = np.matmul(x_translate,K)
    final = np.dot(tempo,vec_x)

    return final[0][0]

def __find_best_worst_matching(K,M):

    best_matching = None
    best_score = None

    worst_matching = None
    worst_score = None

    for matching in M:
        score = __get_cost_matching(K,matching)

        if best_score is None or score < best_score:
            best_score = score
            best_matching = matching

        if worst_score is None or score > worst_score:
            worst_score = score
            worst_matching = matching

    return (best_score, best_matching, worst_score, worst_matching)


def __mesure_difference_nodes(n_model, n_test, evaluation_variable):

    n_model_weight = n_model[evaluation_variable]
    n_test_weight = n_test[evaluation_variable]

    somme = 0
    for i in range(0,len(n_model_weight)):
        #somme += math.pow(n_model_weight[i] - n_test_weight[i],2)
        somme += math.fabs(n_model_weight[i] - n_test_weight[i])
    res = math.sqrt(somme)
    return res

def __mesure_difference_edges(e1, e2):
    return math.sqrt(math.fabs(e1-e2))
    #return math.sqrt(math.pow(e1 - e2, 2))


"""
PERMUTATIONS
"""

class __Permutation(Enum):
    Impossible = -1
    Unclear = 0
    Valid = 1

def __get_cols_permutations(array_test, nb_classes):
    
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

def __check_if_permutation_already_exists(results, perm_to_add):
    
    flag_already_exist = False

    for res in results:
        if np.allclose(res, perm_to_add):
            flag_already_exist = True
    
    return flag_already_exist

def __get_permutation_state(test, x, y):
    
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
        return __Permutation.Valid
    elif value == min(x, y) and flag_one_match == False: 
        return __Permutation.Impossible
    else:
        return __Permutation.Unclear

def __get_permutations_possibilities(nodes_data, classes):
    
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

    temp = __get_cols_permutations(initial_regions_labels, len(classes))

    M = []

    for permu in temp:
        M.append(np.asarray(permu).transpose())

    return M