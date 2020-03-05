import os
import math
import shutil
import random

DATA_TRAIN_DIRECTORY = './Datasets/data_face_all'
DATA_VAL_DIRECTORY = './Datasets/data_face_all'

PATH_TO_TESTS = './Sub_tests'

ELEMENTS_TO_COPY_TRAIN = ['Train_Labels','Train_RGB']
ELEMENTS_TO_COPY_VAL = ['Val_Labels','Val_RGB']

IMAGES_IN_ITERATIONS = [20, 15, 10]

actual_path = os.getcwd()

list_file_train = os.listdir(DATA_TRAIN_DIRECTORY + "/Train_Labels")
list_file_val = os.listdir(DATA_TRAIN_DIRECTORY + "/Val_Labels")

if os.path.isdir(PATH_TO_TESTS):
    shutil.rmtree(PATH_TO_TESTS)

os.mkdir(PATH_TO_TESTS)

for i in IMAGES_IN_ITERATIONS:

    percentage = math.floor(100 * i / len(list_file_train))
    iteration_rule = len(list_file_train) - i

    # Create directory
    dir_path = os.path.join(PATH_TO_TESTS, str(percentage))
    os.mkdir(dir_path)

    if i == max(IMAGES_IN_ITERATIONS):
        iteration_path = os.path.join(dir_path, str(0))
        os.mkdir(iteration_path)

        for copy_path in ELEMENTS_TO_COPY_TRAIN:
            d_path = os.path.join(iteration_path, copy_path)
            os.mkdir(d_path)
            for image_name in list_file_train:
                im_ori_path = os.path.join(DATA_TRAIN_DIRECTORY, copy_path, image_name)
                im_new_path = os.path.join(d_path, image_name)
                shutil.copyfile(im_ori_path, im_new_path)

        for copy_path in ELEMENTS_TO_COPY_VAL:
            d_path = os.path.join(iteration_path, copy_path)
            os.mkdir(d_path)
            for image_name in list_file_val:
                im_ori_path = os.path.join(DATA_VAL_DIRECTORY, copy_path, image_name)
                im_new_path = os.path.join(d_path, image_name)
                shutil.copyfile(im_ori_path, im_new_path)

    else:
        for j in range(0, iteration_rule):
            iteration_path = os.path.join(dir_path, str(j))
            os.mkdir(iteration_path)

            random.shuffle(list_file_train)
            random.shuffle(list_file_val)

            val_images_to_copy = math.floor(percentage/100 * len(list_file_val))

            list_file_train_rand = list_file_train[:i]
            list_file_val_rand = list_file_val[:val_images_to_copy]


            for copy_path in ELEMENTS_TO_COPY_TRAIN:
                d_path = os.path.join(iteration_path, copy_path)
                os.mkdir(d_path)
                for image_name in list_file_train_rand:
                    im_ori_path = os.path.join(DATA_TRAIN_DIRECTORY, copy_path, image_name)
                    im_new_path = os.path.join(d_path, image_name)
                    shutil.copyfile(im_ori_path, im_new_path)

            for copy_path in ELEMENTS_TO_COPY_VAL:
                d_path = os.path.join(iteration_path, copy_path)
                os.mkdir(d_path)
                for image_name in list_file_val_rand:
                    im_ori_path = os.path.join(DATA_VAL_DIRECTORY, copy_path, image_name)
                    im_new_path = os.path.join(d_path, image_name)
                    shutil.copyfile(im_ori_path, im_new_path)

print("ok")