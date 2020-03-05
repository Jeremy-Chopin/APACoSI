import os
import math
import shutil
import random
from Helper_files import helper_subset
import sys
import numpy as np


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
SAVE_PATH = "Results/Mean"
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
        try:
            iteration_path = os.path.join(dir_path, str(0))
            csv_pixel_path = os.path.join(iteration_path, "results_pixel.csv")
            csv_box_path = os.path.join(iteration_path, "results_box.csv")

            res_pixel = np.genfromtxt(csv_pixel_path, delimiter=",", dtype=np.float, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))
            res_box = np.genfromtxt(csv_box_path, delimiter=",", dtype=np.float, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))

            save_path = os.path.join(SAVE_PATH, str(i))
            np.savetxt(save_path + "_pixel.csv", res_pixel, delimiter = ",")
            np.savetxt(save_path + "_box.csv", res_box, delimiter = ",")
        
        except:
            strin = "Error on subdataset : " + iteration_path + "\nMessage : " + str(sys.exc_info()[0])+"\n\n"
            log += strin
            
    else:
        index = 0
        for j in range(0, iteration_rule):
            try:
                iteration_path = os.path.join(dir_path, str(j))
                iteration_path = os.path.join(dir_path, str(0))
                csv_pixel_path = os.path.join(iteration_path, "results_pixel.csv")
                csv_box_path = os.path.join(iteration_path, "results_box.csv")

                res_pixel = np.genfromtxt(csv_pixel_path, delimiter=",", dtype=np.float, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))
                res_box = np.genfromtxt(csv_box_path, delimiter=",", dtype=np.float, skip_header=1, usecols=(1,2,3,4,5,6,7,8,9))

                if j == 0:
                    res_mean_pixel = res_pixel
                    res_mean_box = res_box
                else:
                    res_mean_pixel += res_pixel
                    res_mean_box += res_box
                index +=1
            except:
                strin = "Error on subdataset : " + iteration_path + "\nMessage : " + str(sys.exc_info()[0])+"\n\n"
                log += strin
        
        res_mean_box = res_mean_box / index
        res_mean_pixel = res_mean_pixel / index

        save_path = os.path.join(SAVE_PATH, str(i))
        np.savetxt(save_path + "_pixel.csv", res_mean_pixel, delimiter = ",")
        np.savetxt(save_path + "_box.csv", res_mean_box, delimiter = ",")

if len(log) > 0:
    f = open("logs_csv.txt", "w")
    f.write(log)
    f.close()
    print("Terminé, erreurs sauvegardées dans le fichier logs_csv.txt")
else:
    print("Terminé sans problèmes majeurs")