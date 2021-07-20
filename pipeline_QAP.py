from src.Ar_constructor import Ar_constructor
from src.Ke_constructor import Ke_constructor
from src.Kv_constructor import Kv_constructor
from src.parameters_constructor import Parameters_constructor
from src.refinement_strategy import refinement_strategy_constructor

from src import Graphs
from src import QAP
from src import utils
from src import refinement as ref

def pipeline_QAP(specifier, alpha, pr_mask, image_cnn, nb_classes, Am, max_node_matching =1, max_node_refinement = 2):

    # CREATION DU GRAPHE MODEL - Gm

    nodes_model = utils.create_nodes_model(nb_classes)

    # CREATION DU GRAPHE IMAGE - Gi

    labelled_image, regions, nodes_matching, nodes_refinement = Graphs.define_nodes(image_cnn, pr_mask, max_node_matching, max_node_refinement)

    Ar = Ar_constructor().construct_Ar(specifier, labelled_image, regions, nodes_matching)

    # QAP
    parameters = Parameters_constructor().construct_Parameters(specifier, labelled_image, regions, nodes_matching)

    # Matrice des dissimilarités
    Ke = Ke_constructor().construct_Ke(specifier, Am, Ar, parameters)
    Kv = Kv_constructor().construct_Kv(specifier, nodes_model, nodes_matching, Am, Ar)    

    # Permutations à tester en se basant sur la classe principal des composantes connexes
    M = QAP.define_permutations(regions, nodes_matching, nb_classes)

    # Résultats ordonnées des permutations et les scores associés
    S, Matches = QAP.apply_QAP(Kv, Ke, M, alpha)

    #MATCHING - Matched nodes

    # Obtention des noeuds matchés et de ce qui sont à corriger
    _, nodes_unmatched = utils.get_matching_elements_nodes_3d(Matches[0], nodes_matching)

    # Raffinement

    # Etape 1 : Initialisation des variables

    score_non_regression, Ar_initial, An_initial, matching_initial = ref.change_variable_and_check(specifier, nb_classes, labelled_image, Matches[0], S[0], Am, alpha, pr_mask, parameters, nodes_matching, regions)

    matching_image = ref.create_images_from_ids(labelled_image, matching_initial, regions)

    # Etape 3 : Merging des noeuds

    list_node_to_refined = []

    for node in nodes_unmatched:
        list_node_to_refined.append(node.ids)
    
    for node in nodes_refinement:
        if regions[node.ids - 1].area > 2:
            list_node_to_refined.append(node.ids)

    #merging = ref.apply_refinement_brain(specifier, labelled_image, regions, matching_initial, Am, Ar_initial, An_initial, list_node_to_refined, score_non_regression, pr_mask, nb_classes, parameters, alpha)
    merging = refinement_strategy_constructor().refinement(specifier, labelled_image, regions, matching_initial, Am, Ar_initial, An_initial, list_node_to_refined, score_non_regression, pr_mask, nb_classes, parameters, alpha)

    return labelled_image, regions,matching_initial, merging