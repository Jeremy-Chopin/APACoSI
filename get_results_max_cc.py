import os 
import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
from src import metrics
import pandas as  pd

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

percentage = 50

iteration = 1

Path_to_patients = os.path.join("sub_datasets_brain", str(percentage), str(iteration), "results", "edt")

nb_patients = len(os.listdir(os.path.join("sub_datasets_brain", str(percentage), str(iteration), "data", "RGB_test")))

for i in range(0, nb_patients):
    try:
        patient_path = os.path.join(Path_to_patients, "P" + str(i))

        Path_to_GT_image  = os.path.join(patient_path, str(i) + "_gt.nii.gz")
        Path_to_cnn_image  = os.path.join(patient_path, str(i) + "_cnn.nii.gz")
        Path_to_proposal_image = os.path.join(patient_path, str(i) + "_before_filter.nii.gz")

        nifti_gt = nib.load(Path_to_GT_image)
        img_gt = nifti_gt.get_fdata().astype(np.int32)

        nifti_cnn = nib.load(Path_to_cnn_image)
        img_cnn = nifti_cnn.get_fdata().astype(np.int32)

        nifti_proposal = nib.load(Path_to_proposal_image)
        img_proposal = nifti_proposal.get_fdata().astype(np.int32)


        img_cnn_cc = get_largest_cc_results(img_cnn, 14)

        img_proposal_cc = get_largest_cc_results(img_proposal, 14)

        datas = ["dice", "precision", "recall", "hausdorff", "nb_CC"]

        score_cnn_CC = metrics.all_informations_data_frame(datas, img_cnn_cc, img_gt)

        score_proposal_CC = metrics.all_informations_data_frame(datas, img_proposal_cc, img_gt)

        result = pd.concat([score_cnn_CC, score_proposal_CC], axis=1, join="inner")

        print(result)

        result_path = os.path.join(patient_path, "result_CC.csv")
        result.to_csv(result_path)

        print("ok")
    except:
        pass