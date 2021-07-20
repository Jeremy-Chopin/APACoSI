import time
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import label
from src import Graphs
from src import QAP
from src import utils
from src import metrics
from src import refinement as ref

import os
import numpy as np
import pandas as pd
from scipy import ndimage
import nibabel as nib
from skimage.measure import label, regionprops
import pipeline_QAP

# Configuration des chemins

def get_largest_cc_results(image_cnn, nb_classes):

    image = np.zeros(image_cnn.shape)

    for i in range(1, nb_classes+1):
        mask = np.where(image_cnn == i, 1, 0).astype(np.bool)
        lbl = label(mask, connectivity=2)
        regions = regionprops(lbl)
        biggest_region = None
        for region in regions:
            if biggest_region is None:
                biggest_region = region
            else:
                if biggest_region.area < region.area:
                    biggest_region = region
        image = np.where(lbl == biggest_region.label, i, image)

    return image


def save_img_to_nii(img, path):

    nib_image = nib.Nifti1Image(img, np.eye(4))

    nib_image.header.get_xyzt_units()
    nib_image.to_filename(path)


percentage = 100

iteration = 0

KNOWLEDGES_PATH = os.path.join(".", "sub_datasets_brain", str(percentage), str(iteration),  "knowledges", "edt.npy")

SEGMENTATION_DIRECTORY = os.path.join(".", "sub_datasets_brain", str(percentage), str(iteration), "segmentation")

RESULTS_DIRECTORY = os.path.join(".", "sub_datasets_brain", str(percentage), str(iteration), "results")

# Configuration des parametres

alpha = 0.5

nb_classes = 14

specifier = "edt_min"

max_node_matching = 2
max_node_refinement = np.inf

# Creation des chemins

if os.path.isdir(RESULTS_DIRECTORY) == False:
    os.mkdir(RESULTS_DIRECTORY)

RESULTS_DIRECTORY_specifier = os.path.join(".", "sub_datasets_brain", str(percentage), str(iteration), "results", specifier)

if os.path.isdir(RESULTS_DIRECTORY_specifier) == False:
    os.mkdir(RESULTS_DIRECTORY_specifier)

# Methodes

nb_files = int(len(os.listdir(SEGMENTATION_DIRECTORY)) / 4)

times = []
errors = []

for k in range(0, nb_files):

    t_initial = time.time()

    try:
        patient_path = os.path.join(RESULTS_DIRECTORY_specifier, "P" + str(k))
        if os.path.isdir(patient_path) == False:
            os.mkdir(patient_path)

        """""""""""""""""""""""""""""""""""""""
            Chargement des données
        """""""""""""""""""""""""""""""""""""""

        pr_mask = np.load(os.path.join(SEGMENTATION_DIRECTORY, str(k) + "pr.npy"))
        gt_mask = np.load(os.path.join(SEGMENTATION_DIRECTORY, str(k) + "gt.npy"))
        image_cnn = np.load(os.path.join(SEGMENTATION_DIRECTORY, str(k) + "argmax.npy"))
        intensity_input = np.load(os.path.join(SEGMENTATION_DIRECTORY, str(k) + "intensity.npy"))

        dims = len(image_cnn.shape)

        """""""""""""""""""""""""""""""""""""""
            Etapes de pre-traitements
        """""""""""""""""""""""""""""""""""""""

        pr_mask = np.transpose(pr_mask, (1,2,3,0))
        
        image_cnn = np.where(intensity_input > 0, image_cnn, 0)

        image_cnn = ndimage.median_filter(image_cnn, 3)

        pr_mask = pr_mask[:,:,:, 1:nb_classes + 1]

        Am= Graphs.load_structural_model(KNOWLEDGES_PATH, nb_classes)

        #Am = np.transpose(Am, (2,0,1))

        #Am = np.transpose(Am, (1,0,2))

        labelled_image, regions, matching_inter, merging_final = pipeline_QAP.pipeline_QAP(specifier, alpha, pr_mask, image_cnn, nb_classes, Am, max_node_matching, max_node_refinement)
        
        merging_final_filtre, distance_recap = ref.correct_merging_distance(labelled_image, regions, matching_inter, merging_final, 20)

        # Etape 4 : Obtention des images

        biggest_CC_image = get_largest_cc_results(image_cnn, nb_classes)

        matching_image = ref.create_images_from_ids(labelled_image, matching_inter, regions)

        proposal_image = ref.create_images_from_ids(labelled_image, merging_final, regions)

        post_refinement_image_CC = get_largest_cc_results(proposal_image, nb_classes)

        image_refinement_distance = ref.create_images_from_ids(labelled_image, merging_final_filtre, regions)

        #image_refinement_distance = median_filter(image_refinement_distance, 3)

        # Etape 4 : Sauvegardes des images

        cnn_output_path = os.path.join(patient_path, str(k) + "_cnn_output.nii.gz")
        save_img_to_nii(image_cnn, cnn_output_path)

        gt_path = os.path.join(patient_path, str(k) + "_gt.nii.gz")
        save_img_to_nii(gt_mask, gt_path)

        cnn_CC = os.path.join(patient_path, str(k) + "_cnn_output_CC.nii.gz")
        save_img_to_nii(biggest_CC_image, cnn_CC)
        
        proposal_path = os.path.join(patient_path, str(k) + "_proposal.nii.gz")
        save_img_to_nii(proposal_image, proposal_path)

        post_processing_path = os.path.join(patient_path, str(k) + "_proposal_CC.nii.gz")
        save_img_to_nii(post_refinement_image_CC, post_processing_path)

        """""""""""""""""""""""""""""""""""""""
                    Metrics
        """""""""""""""""""""""""""""""""""""""

        datas = ["dice", "precision", "recall", "hausdorff", "nb_CC"]

        score_cnn = metrics.all_informations_data_frame(datas, image_cnn, gt_mask)

        score_cnn_CC = metrics.all_informations_data_frame(datas, biggest_CC_image, gt_mask)

        score_before_refinement = metrics.all_informations_data_frame(datas, matching_image, gt_mask)

        score_after_refinement = metrics.all_informations_data_frame(datas, proposal_image, gt_mask)

        score_post_refinement_CC = metrics.all_informations_data_frame(datas, post_refinement_image_CC, gt_mask)

        result = pd.concat([score_cnn, score_cnn_CC, score_before_refinement, score_after_refinement, score_post_refinement_CC], axis=1, join="inner")

        print(result)

        result_path = os.path.join(patient_path, "result.csv")
        result.to_csv(result_path)

        processing_time = time.time() - t_initial
        times.append(processing_time)

    except:
        print("error on file" + str(k))
        errors.append(patient_path)

with open(os.path.join(RESULTS_DIRECTORY_specifier, "error_log.txt"), "w") as output:
    output.write(str(errors))

times = np.asarray(times)
np.savetxt(os.path.join(RESULTS_DIRECTORY_specifier, "processing_time.csv"), times)