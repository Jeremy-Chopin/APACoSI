from Helper_files import helper
from Helper_files import helper_QAP as QAP
from Helper_files import helper_segmentation as SEG

from Helper_files import helper_plot as PLOT
from Helper_files import helper_corrections as CORREC
from Helper_files import helper_main as MAIN

import cv2
import numpy as np
import networkx as nx
import string
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from operator import add, mul
from scipy.signal import medfilt
import sys


# Définitions des constantes

DATA_TEST_DIRECTORY_RGB = './Datasets/data_face_all/TEST_RGB'
DATA_TEST_DIRECTORY_Labels = './Datasets/data_face_all/TEST_Labels'

CLASSES = ['background', 'hair', 'face', 'l_brow', 'r_brow', 'l_eye','r_eye', 'nose','mouth']

model_path = 'Sub_tests/100/0/best_model.h5'
KNOWLEDGES_PATH = "Knowledges/face_elements_transform_distances.csv" 


MAIN.test_refinement_on_dataset(model_path, CLASSES, DATA_TEST_DIRECTORY_RGB, DATA_TEST_DIRECTORY_Labels, KNOWLEDGES_PATH, 0.4)