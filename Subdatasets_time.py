import os
import math
import shutil
import random
#from Helper_files import helper_subset
#from Helper_files import helper_main as MAIN
import sys
import numpy as np
import statistics

def get_time_data_on_datasets(path):

    list_dir = os.listdir(path)

    time_data = []

    precedent_time = 0
    actual_time = 0
    for i in range(0, len(list_dir)):
        image_path = os.path.join(path, list_dir[i])
        if i == 0:
            precedent_time = os.path.getctime(image_path)
        else:
            actual_time = os.path.getctime(image_path)
            execution_time = actual_time - precedent_time
            time_data.append(execution_time)
            precedent_time = actual_time
    return time_data

def get_np_array_data(time_data):

    data = np.zeros((1, 5))

    data[0][0] = mean = statistics.mean(time_data)
    data[0][1] = stdev = statistics.stdev(time_data)
    data[0][2] = median = statistics.median(time_data)
    data[0][3] = tmin = min(time_data)
    data[0][4] = tmax = max(time_data)

    return data

def save_time_data_csv(csv_path, time_data):

    meanings = "Mean, Standard deviation, Median, Min, Max"
    np.savetxt(csv_path, time_data, delimiter = ",", header = meanings)


PATH_TO_TESTS = './Sub_tests'

IMAGES_IN_ITERATIONS = [20, 15, 10, 5]

actual_path = os.getcwd()

log = ""

DATA_TRAIN_DIRECTORY = './Datasets/data_face_all'
list_file_train = os.listdir(DATA_TRAIN_DIRECTORY + "/Train_Labels")

full_data = None

for i in IMAGES_IN_ITERATIONS:

        percentage = math.floor(100 * i / len(list_file_train))
        iteration_rule = 20

        # Create directory
        dir_path = os.path.join(PATH_TO_TESTS, str(percentage))

        if i == max(IMAGES_IN_ITERATIONS):
            iteration_path = os.path.join(dir_path, str(0))
            tdata = get_time_data_on_datasets(os.path.join(iteration_path, "Images"))
            tdata_np = get_np_array_data(tdata)
            full_data = tdata_np
            tdata = None

        else:
            for j in range(0, iteration_rule):
                    try:
                        iteration_path = os.path.join(dir_path, str(j))
                        if tdata == None:
                            tdata = get_time_data_on_datasets(os.path.join(iteration_path, "Images"))
                        else:
                            for time in get_time_data_on_datasets(os.path.join(iteration_path, "Images")):
                                tdata.append(time)
                        
                    except:
                        strin = "Error on subdataset : " + iteration_path + "\nMessage : " + str(sys.exc_info()[0])+"\n\n"
                        log += strin
            
            tdata_np = get_np_array_data(tdata)
            if full_data is None:
                full_data = tdata_np
            else:
                full_data = np.concatenate((full_data, tdata_np), axis=0)
            tdata = None


save_time_data_csv(os.path.join("Results", "Time", "time_data.csv"), full_data)
print(full_data)