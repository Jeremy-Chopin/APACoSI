import os
import math
import shutil
import random
from Helper_files import helper_subset
from Helper_files import helper_main as MAIN
import sys


DATA_TRAIN_DIRECTORY = './Datasets/data_face_all'
DATA_VAL_DIRECTORY = './Datasets/data_face_all'

DATA_TEST_DIRECTORY_RGB = './Datasets/data_face_all/TEST_RGB'
DATA_TEST_DIRECTORY_Labels = './Datasets/data_face_all/TEST_Labels'

PATH_TO_TESTS = './Sub_tests'

ELEMENTS_TO_COPY_TRAIN = ['Train_Labels','Train_RGB']
ELEMENTS_TO_COPY_VAL = ['Val_Labels','Val_RGB']

IMAGES_IN_ITERATIONS = [20, 15, 10, 5]

CLASSES = ['background', 'hair', 'face', 'l_brow', 'r_brow', 'l_eye','r_eye', 'nose','mouth']
KNOWLEDGES_PATH = "Knowledges/face_elements_transform_distances.csv" 

actual_path = os.getcwd()

list_file_train = os.listdir(DATA_TRAIN_DIRECTORY + "/Train_Labels")
list_file_val = os.listdir(DATA_TRAIN_DIRECTORY + "/Val_Labels")

log = ""

for i in IMAGES_IN_ITERATIONS:
        percentage = math.floor(100 * i / len(list_file_train))
        iteration_rule = 20

        # Create directory
        dir_path = os.path.join(PATH_TO_TESTS, str(percentage))

        if i == max(IMAGES_IN_ITERATIONS):
            iteration_path = os.path.join(dir_path, str(0))
            model_path = os.path.join(iteration_path, "best_model.h5")
            MAIN.test_on_dataset(model_path, CLASSES, DATA_TEST_DIRECTORY_RGB, DATA_TEST_DIRECTORY_Labels, iteration_path, KNOWLEDGES_PATH, 0.6)
                
        else:
            for j in range(0, iteration_rule):
                    try:
                        iteration_path = os.path.join(dir_path, str(j))
                        model_path = os.path.join(iteration_path, "best_model.h5")
                        MAIN.test_on_dataset(model_path, CLASSES, DATA_TEST_DIRECTORY_RGB, DATA_TEST_DIRECTORY_Labels, iteration_path, KNOWLEDGES_PATH, 0.6)
                    except:
                        strin = "Error on subdataset : " + iteration_path + "\nMessage : " + str(sys.exc_info()[0])+"\n\n"
                        log += strin

if len(log) > 0:
    f = open("logs.txt", "w")
    f.write(log)
    f.close()
    print("Terminé, erreurs sauvegardées dans le fichier logs.txt")
else:
    print("Terminé sans problèmes majeurs")