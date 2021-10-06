# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import time
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.measure import label, regionprops
from scipy.ndimage.filters import median_filter
import pickle
from matplotlib import pyplot as plt
import cv2

from src import Graphs
from src import QAP
from src import utils
from src import metrics
from src import refinement as ref
import pipeline_QAP

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

def load_image_from_dict(image_lbl_path, dict_path, nb_classes):
    loaded_image = nib.load(image_lbl_path).get_fdata()
    with open(dict_path, 'rb') as handle:
        b = pickle.load(handle)

    dims = list(loaded_image.shape)

    dims.append(nb_classes+1)

    base_image = np.zeros(dims)

    uni = np.unique(loaded_image)

    for value in uni:

        region_mask = np.where(loaded_image == value, 1, 0)

        if len(loaded_image.shape) > 2:
            proba_mask = np.expand_dims(region_mask, axis=3) * b[value]
        else:
            proba_mask = np.expand_dims(region_mask, axis=2) * b[value]

        base_image = base_image + proba_mask
    return base_image

# PATHs
#specifier = "centroid"
specifier = "edt_signed"

DATA_PATH = os.path.join(".", "DATA", "2D")

KNOWLEDGES_PATH = os.path.join(DATA_PATH, "Knowledges", specifier + ".npy")

RESULT_PATH = os.path.join(".", "RESULTS", "2D", "result_" + specifier + ".png")

# Parameters
alpha = 0.5
nb_classes = 8

max_node_matching = 2
max_node_refinement = np.inf

t_initial = time.time()

intensity_input = cv2.imread(os.path.join(DATA_PATH, "9.png"))
intensity_input = cv2.cvtColor(intensity_input, cv2.COLOR_BGR2RGB)

pr_mask = np.load(os.path.join(DATA_PATH, "pr.npy"))
gt_mask = np.load(os.path.join(DATA_PATH, "gt.npy"))
gt_image = np.argmax(gt_mask, axis = 2)

image_cnn = np.argmax(pr_mask, axis=len(pr_mask.shape) - 1)

dims = len(image_cnn.shape)

"""""""""""""""""""""""""""""""""""""""
    Etapes de pre-traitements
"""""""""""""""""""""""""""""""""""""""

pr_mask = pr_mask[:,:,1:nb_classes + 1]

image_cnn = ndimage.median_filter(image_cnn, 3).astype(np.int32)

Am= Graphs.load_structural_model(KNOWLEDGES_PATH, nb_classes)

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

plt.subplot(2,2,1); plt.title("Original"); plt.imshow(intensity_input)
plt.subplot(2,2,2); plt.title("Expert"); plt.imshow(gt_image)
plt.subplot(2,2,3); plt.title("CNN"); plt.imshow(image_cnn)
plt.subplot(2,2,4); plt.title("Proposal"); plt.imshow(proposal_image)

plt.show()

plt.savefig(RESULT_PATH)

plt.close()

processing_time = time.time() - t_initial

print("Image processed in : " + str(processing_time) + " s")