from Helper_files import helper
from Helper_files import helper_QAP as QAP
from Helper_files import helper_segmentation as SEG
from Helper_files import helper_plot as PLOT
from Helper_files import helper_corrections as CORREC
from Helper_files import helper_graphe as GRAPHE

from progress.bar import Bar
import cv2
import numpy as np
import networkx as nx
from skimage.measure import label, regionprops
from operator import add, mul
from matplotlib import pyplot as plt
from scipy.signal import medfilt
import shutil
import skimage
import sys
import os

def test_on_dataset(model_path, classes, rgb_directory_path, labels_directory_path, results_directory_path, knowledges_path, alpha):

    save_imgs_path = os.path.join(results_directory_path, "Images")

    if os.path.isdir(save_imgs_path):
        shutil.rmtree(save_imgs_path)

    os.mkdir(save_imgs_path)

    dataset = SEG.load_dataset(rgb_directory_path, labels_directory_path, classes)
    model = SEG.load_model(model_path, classes)

    somme = 0
    res_non_corrected_m = []
    res_corrected_m = []

    bar = Bar('Processing dataset : ' + results_directory_path, max = len(dataset.ids))
    for dataset_index in range(0, len(dataset.ids)):
        try:
            
            """""""""""""""""""""""""""""""""""""""""""""
                CHARGEMENT DE L'IMAGE ET SEGMENTATION
            """""""""""""""""""""""""""""""""""""""""""""
            image_name = dataset.ids[dataset_index].split('.')[0]
            image, gt_mask = dataset[dataset_index]

            pr_mask = SEG.get_segmentation(image, model)
            image_gray = SEG.reduce_pmap_depth_confusion(pr_mask, 20, 0.1)
            gt_gray = SEG.reduce_pmap_depth(gt_mask)

            """""""""""""""""""""""""""""""""
                AJOUT D'UN FILTRE MEDIAN
            """""""""""""""""""""""""""""""""
            
            image_filter = medfilt(image_gray, 9)
            image_gray = image_filter

            """""""""""""""""""""""""""""""""""""""
                CREATION DU GRAPHE MODELE - Gm
            """""""""""""""""""""""""""""""""""""""
            
            nth_dims_image_model = GRAPHE.create_model_nth_images(len(classes))

            labels_name_image_model = GRAPHE.create_model_labels(len(classes))

            Gm = GRAPHE.create_graphe_model_from_csv(knowledges_path, labels_name_image_model, nth_dims_image_model)
            
            """""""""""""""""""""""""""""""""""""""
                CREATION DU GRAPHE IMAGE - Gi
            """""""""""""""""""""""""""""""""""""""

            labelled_image, Gi, nodes_data, labels_name_image_test, regions = GRAPHE.create_graph_image(image_gray, pr_mask)

            labelled_image_conf, Gi_conf, nodes_data_conf, labels_name_image_test_conf, regions_conf = GRAPHE.create_graph_image_wout_confusion(image_gray, pr_mask, confusion_seuil=20)
            """""""""""""""""""""""""""""""""""""""
                            QAP
            """""""""""""""""""""""""""""""""""""""

            best_score, best_matching, K = QAP.apply_QAP_with_refinment(Gi, Gm, labels_name_image_test, labels_name_image_model, nodes_data, classes, alpha)

            match_model, match_test = QAP.get_matching_elements_labels(best_matching, labels_name_image_model, labels_name_image_test)

            Ai = Ai=nx.adjacency_matrix(Gi,nodelist=labels_name_image_test).toarray()

            final_matching = QAP.refinment_matching_third(K, best_matching, regions,nodes_data, match_test, Ai, alpha)

            """""""""""""""""""""""""""""""""""""""
                        CORRRECTIONS
            """""""""""""""""""""""""""""""""""""""

            gd = cv2.imread(labels_directory_path + '/' + dataset.ids[dataset_index])
            gd = cv2.cvtColor(gd, cv2.COLOR_BGR2GRAY)

            image_rgb = cv2.imread(rgb_directory_path + '/' + dataset.ids[dataset_index])
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            image_gray = image_gray + 1
            
            #corrected = CORREC.non_matching_regions(regions, nodes_data, labelled_image, match_test)
            corrected = CORREC.non_matching_regions_refinment(regions, labelled_image, final_matching)

            plt.figure()
            plt.subplot(2,3,1); plt.axis('off'); plt.title('RGB'); plt.imshow(image_rgb)
            plt.subplot(2,3,2); plt.axis('off'); plt.title('seg Expt'); plt.imshow(gt_gray)
            plt.subplot(2,3,4); plt.axis('off'); plt.title('seg U-net'); plt.imshow(image_gray)
            plt.subplot(2,3,5); plt.axis('off'); plt.title('S_pixels'); plt.imshow(labelled_image)
            plt.subplot(2,3,6); plt.axis('off'); plt.title('Correc'); plt.imshow(corrected)
            plt.savefig(save_imgs_path + "/" + image_name + ".png")
            plt.close()

            res_non_corrected, res_non_corrected_box = CORREC.get_iou(gt_gray, image_gray, len(classes), classes)
            res_corrected, res_corrected_box = CORREC.get_iou(gt_gray, corrected, len(classes), classes)

            if len(res_corrected_m) != 0:
                res_non_corrected_m = list( map(add, res_non_corrected_m, res_non_corrected))
                res_corrected_m = list( map(add, res_corrected_m, res_corrected))
                res_non_corrected_box_m = list( map(add, res_non_corrected_box_m, res_non_corrected_box))
                res_corrected_box_m = list( map(add, res_corrected_box_m, res_corrected_box))
            else:
                res_non_corrected_m = res_non_corrected
                res_corrected_m = res_corrected
                res_non_corrected_box_m = res_non_corrected_box
                res_corrected_box_m = res_corrected_box
            
            somme += 1
        except:
            pass
            #print("Unexpected error on image : " + str(dataset.ids[dataset_index]) +"\n", sys.exc_info()[0])
        finally:
            bar.next()
    bar.finish()

    s = []

    for i in range(0, len(res_corrected)):
        s.append(1/somme)

    res_non_corrected_m = list( map(mul, res_non_corrected_m, s))
    res_corrected_m = list( map(mul, res_corrected_m, s))

    res_non_corrected_box_m = list( map(mul, res_non_corrected_box_m, s))
    res_corrected_box_m = list( map(mul, res_corrected_box_m, s))

    res_path_pixel = os.path.join(results_directory_path, "results_pixel.csv")
    res_path_box = os.path.join(results_directory_path, "results_box.csv")

    CORREC.create_csv(res_non_corrected_m, res_corrected_m, res_path_pixel, classes)
    CORREC.create_csv(res_non_corrected_box_m, res_corrected_box_m, res_path_box, classes)

def test_refinement_on_dataset(model_path, classes, rgb_directory_path, labels_directory_path, knowledges_path, alpha):

    dataset = SEG.load_dataset(rgb_directory_path, labels_directory_path, classes)
    model = SEG.load_model(model_path, classes)

    for dataset_index in range(0, len(dataset.ids)):
        try:
            
            """""""""""""""""""""""""""""""""""""""""""""
                CHARGEMENT DE L'IMAGE ET SEGMENTATION
            """""""""""""""""""""""""""""""""""""""""""""
            image_name = dataset.ids[dataset_index].split('.')[0]
            image, gt_mask = dataset[dataset_index]

            pr_mask = SEG.get_segmentation(image, model)
            image_gray = SEG.reduce_pmap_depth(pr_mask)
            gt_gray = SEG.reduce_pmap_depth(gt_mask)

            """""""""""""""""""""""""""""""""
                AJOUT D'UN FILTRE MEDIAN
            """""""""""""""""""""""""""""""""
            
            image_filter = medfilt(image_gray, 9)
            image_gray = image_filter

            """""""""""""""""""""""""""""""""""""""
                CREATION DU GRAPHE MODELE - Gm
            """""""""""""""""""""""""""""""""""""""
            
            nth_dims_image_model = GRAPHE.create_model_nth_images(len(classes))

            labels_name_image_model = GRAPHE.create_model_labels(len(classes))

            Gm = GRAPHE.create_graphe_model_from_csv(knowledges_path, labels_name_image_model, nth_dims_image_model)
            
            """""""""""""""""""""""""""""""""""""""
                CREATION DU GRAPHE IMAGE - Gi
            """""""""""""""""""""""""""""""""""""""

            labelled_image, Gi, nodes_data, labels_name_image_test, regions = GRAPHE.create_graph_image(image_gray, pr_mask)

            """""""""""""""""""""""""""""""""""""""
                            QAP
            """""""""""""""""""""""""""""""""""""""

            best_score, best_matching, K = QAP.apply_QAP_with_refinment(Gi, Gm, labels_name_image_test, labels_name_image_model, nodes_data, classes, alpha)

            match_model, match_test = QAP.get_matching_elements_labels(best_matching, labels_name_image_model, labels_name_image_test)

            Ai = Ai=nx.adjacency_matrix(Gi,nodelist=labels_name_image_test).toarray()

            #final_matching = QAP.refinment_matching(K, best_matching, regions, match_test)
            final_matching = QAP.refinment_matching_third(K, best_matching, regions,nodes_data, match_test, Ai, alpha)

            """""""""""""""""""""""""""""""""""""""
                        CORRRECTIONS
            """""""""""""""""""""""""""""""""""""""

            gd = cv2.imread(labels_directory_path + '/' + dataset.ids[dataset_index])
            gd = cv2.cvtColor(gd, cv2.COLOR_BGR2GRAY)

            image_rgb = cv2.imread(rgb_directory_path + '/' + dataset.ids[dataset_index])
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

            image_gray = image_gray + 1
            
            #corrected = CORREC.non_matching_regions(regions, nodes_data, labelled_image, match_test)
            corrected = CORREC.non_matching_regions_refinment(regions, labelled_image, final_matching)

            plt.figure()
            plt.subplot(1,3,1); plt.title('rgb'); plt.imshow(image_rgb)
            plt.subplot(1,3,2); plt.title('original'); plt.imshow(labelled_image)
            plt.subplot(1,3,3); plt.title('corrections'); plt.imshow(corrected)
            plt.savefig("Results_rf/" + image_name + ".png")
            plt.close()

            #res_non_corrected, res_non_corrected_box = CORREC.get_iou(gt_gray, image_gray, len(classes), classes)
            #res_corrected, res_corrected_box = CORREC.get_iou(gt_gray, corrected, len(classes), classes)

        except:
            pass