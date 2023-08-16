import os
import numpy as np
from PIL import Image

class Subject:
   
    def __init__(self, id, name, groundtruth = None, probability_map = None, segmentation = None, labels = None, K = None, affine = None):
        self.id = id
        self.groundtruth = groundtruth
        self.probability_map = probability_map
        self.segmentation = segmentation
        self.labels = labels
        self.K = K
        self.name = name
        self.affine = affine