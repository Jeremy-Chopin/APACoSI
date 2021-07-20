import os
import numpy as np
import cv2
import nibabel as nib

from knowledges_constructor import knowledges_constructor


def load_file(path):
    if ".png" in path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif ".nii.gz" in path:
        image = nib.load(path)
        image = image.get_fdata()
    else:
        print("Error !")

    return image

def create_adjency_matrix(specifier, label_path, nb_classes):
    
    files = os.listdir(label_path)

    Am_temp = None
    s = 0

    for f in files:
        file_path = os.path.join(label_path, f)
        image = load_file(file_path)
        if Am_temp is None:
            Am_temp = knowledges_constructor().construct_knowledges(specifier, image, nb_classes)
        else:
            Am_temp += knowledges_constructor().construct_knowledges(specifier, image, nb_classes)
        s +=1

    Am = Am_temp / s
    
    return Am

DATASET_PATH = "sub_datasets_brain"

NB_CLASSES = 14

SPECIFIER = "edt"

percentages = os.listdir(DATASET_PATH)

for percentage in percentages:
    percentage_path = os.path.join(DATASET_PATH, percentage)
    
    if os.path.isdir(percentage_path):
        iterations = os.listdir(percentage_path)

        for iteration in iterations:
            iteration_path = os.path.join(percentage_path, iteration)

            if os.path.isdir(iteration_path):

                local_knowlege_path = os.path.join(iteration_path, "knowledges")
                if os.path.isdir(local_knowlege_path) is False:
                    os.mkdir(local_knowlege_path)

                train_labels_path = os.path.join(iteration_path, "data","Labels_train")

                Am = create_adjency_matrix(SPECIFIER, train_labels_path, NB_CLASSES)

                if len(Am.shape) >= 3:
                    Am_path = os.path.join(local_knowlege_path, SPECIFIER + ".npy")
                    np.save(Am_path, Am)
                else:
                    Am_path = os.path.join(local_knowlege_path, SPECIFIER + ".csv")
                    np.savetxt(Am_path, Am)