import os
import math
import shutil
import random
import helper_subset


DATA_TRAIN_DIRECTORY = './Datasets/data_face_all'
DATA_VAL_DIRECTORY = './Datasets/data_face_all'

PATH_TO_TESTS = './Sub_tests'

ELEMENTS_TO_COPY_TRAIN = ['Train_Labels','Train_RGB']
ELEMENTS_TO_COPY_VAL = ['Val_Labels','Val_RGB']

IMAGES_IN_ITERATIONS = [20, 15, 10]

CLASSES = ['background', 'hair', 'face', 'l_brow', 'r_brow', 'l_eye','r_eye', 'nose','mouth']

actual_path = os.getcwd()

list_file_train = os.listdir(DATA_TRAIN_DIRECTORY + "/Train_Labels")
list_file_val = os.listdir(DATA_TRAIN_DIRECTORY + "/Val_Labels")

for i in IMAGES_IN_ITERATIONS:

    percentage = math.floor(100 * i / len(list_file_train))
    iteration_rule = 20

    # Create directory
    dir_path = os.path.join(PATH_TO_TESTS, str(percentage))

    if i == max(IMAGES_IN_ITERATIONS):
        iteration_path = os.path.join(dir_path, str(0))
        model_path = os.path.join(iteration_path, "best_model.h5")
        helper_subset.train_model(iteration_path, CLASSES, model_path, False)
            
    else:
        for j in range(0, iteration_rule):
            iteration_path = os.path.join(dir_path, str(j))
            model_path = os.path.join(iteration_path, "best_model.h5")
            helper_subset.train_model(iteration_path, CLASSES, model_path, False)

print("ok")