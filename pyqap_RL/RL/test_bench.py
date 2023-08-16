from .subjects import Subjects
from .params import Params
from .RL_env import GymQapEnv
from .RL_utils import apply_matching
import numpy as np
import os
import time
import pandas as pd
from progress.bar import Bar
from matplotlib import pyplot as plt
import nibabel as nib
from skimage.measure import regionprops
import math


def calculate_number_permutations(nb_regions, nb_class, seed_size):

    sequential_matching = math.perm(nb_regions, seed_size)

    for i in range(0, nb_class - seed_size):
        sequential_matching += nb_regions - seed_size - i

    return sequential_matching

def save_img_to_nii(img, path, affine):

    nib_image = nib.Nifti1Image(img, affine)

    nib_image.header.get_xyzt_units()
    nib_image.to_filename(path)

class TestBench():

    def __init__(self, subjects : Subjects, parameters : Params, env : GymQapEnv, results_directory_path):
        self.subjects = subjects
        self.parameters = parameters
        self.env = env
        
        self.results_directory_path = results_directory_path
        if os.path.isdir(self.results_directory_path) is False:
            os.mkdir(self.results_directory_path)
        
        self.results_array_path = os.path.join(results_directory_path, 'array')
        if os.path.isdir(self.results_array_path) is False:
            os.mkdir(self.results_array_path)
            
        self.results_image_path = os.path.join(results_directory_path, 'image')
        if os.path.isdir(self.results_image_path) is False:
            os.mkdir(self.results_image_path)

    def test(self, path, nb_class, seed_size):
        
        episodes_time = []
        permutations_numbers = []
        
        bar = Bar('Testing : ', max=len(self.subjects.list_subjects))
        for sub in self.subjects.list_subjects:
            
            regions = regionprops(sub.labels)

            permutations_numbers.append(calculate_number_permutations(len(regions), nb_class,seed_size))

            t0 = time.time()
            s = self.env.reset(sub.labels, sub.groundtruth, sub.K)
            rAll, d = 0, False

            while d is False:
                #Select action
                a = np.argmax(self.parameters.Q[s, :])

                #Get new state and reward from environment
                s1,r,d,_ = self.env.step(a, True)

                #Update current state
                s = s1

            adjacency_matrix  = self.env.get_adjacency_matrix()
            
            matching_image = apply_matching(adjacency_matrix, sub.labels)
            
            self.__save_matching_image(matching_image, sub.name, sub.affine)

            episodes_time.append(time.time() - t0)
            #plt.imshow(matching_image); plt.show()

            bar.next()
        
        if path:
            self.__save_inference_time(episodes_time, permutations_numbers, path)

    def load_Q_table(self, path):
        df = pd.read_csv(path, sep=';')
        q = df.to_numpy()[:,1:]
        self.parameters.Q = q

    def __save_matching_image(self, image, index, affine = None):

        if len(image.shape) == 3:
            #array_path = os.path.join(self.results_array_path, '{}.npy'.format(index))
            #np.save(array_path, image)
            
            image_path = os.path.join(self.results_image_path, '{}.nii.gz'.format(index))

            save_img_to_nii(image, image_path, affine)
            
            """if len(image.shape) == 2:
                plt.imshow(image)
                plt.tight_layout()
                plt.savefig(image_path)
                plt.close()
            else:
                save_img_to_nii(image, image_path)"""
        else:
            array_path = os.path.join(self.results_array_path, '{}.npy'.format(index))
            np.save(array_path, image)
            
            image_path = os.path.join(self.results_image_path, '{}.png'.format(index))
            plt.imshow(image)
            plt.tight_layout()
            plt.savefig(image_path)
            plt.close()
            
        
    def __save_inference_time(self, episodes_time, permutation_number, path):

        files = []
        for i in range(len(self.subjects.list_subjects)):
            files.append(self.subjects.list_subjects[i].name)
        
        episodes_time = np.array(episodes_time)
        permutation_number = np.array(permutation_number)

        columns = ['nb_permutations', 'processing_time']

        a = np.stack((permutation_number, episodes_time), axis=1).astype(np.float32)

        df = pd.DataFrame(a, index=files, columns=columns)
        df.to_csv(path, sep=';')