from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
#from .subjects import Subjects
from .subjects import Subjects
from .params import Params
from .training_parameters import TrainingParameters

from progress.bar import Bar
from .RL_utils import get_random_actions_from_states_dict

import numpy as np
import math
import time
import pandas as pd
from tqdm import tqdm


class TrainBench():

    def __init__(self, subjects : Subjects, parameters : Params, training_parameters : TrainingParameters, env):
        self.subjects = subjects
        self.parameters = parameters
        self.training_parameters = training_parameters
        self.env = env
        

    def train(self, q_table_path, training_time_path = None):
        
        total_episodes = self.parameters.nb_episodes_per_image * self.subjects.nb_subjects

        steps = 0
        episodes_time = []
        eps = self.training_parameters.max_epsilon

        with tqdm(total=total_episodes) as pbar:
            for i in range(self.parameters.nb_episodes_per_image):
                #t1 = time.perf_counter()
                for sub in self.subjects.list_subjects:

                    t0 = time.time()

                    s = self.env.reset(sub.segmentation, sub.groundtruth, sub.K)
                    rAll, d = 0, False

                    for t in range(100):
                        #Select action
                        a = np.argmax(self.parameters.Q[s, :])
                        if (np.random.rand(1)[0] < eps):# or (np.max(self.parameters.Q[s, :]) == 0):
                            a = get_random_actions_from_states_dict(self.parameters.actions_dict, s)

                        #Get new state and reward from environment
                        s1,r,d,_ = self.env.step(a)

                        #print(self.parameters.Q[s,a])
                        #Update Q-Table with new knowledge
                        self.parameters.Q[s,a] = self.parameters.Q[s,a] + self.training_parameters.learning_rate * (r + self.training_parameters.discount_factor * np.max(self.parameters.Q[s1,:]) - self.parameters.Q[s,a])
                        #print(self.parameters.Q[s,a])
                        #Overal reward
                        rAll += r

                        #Update current state
                        s = s1

                        
                        if d: #Si on termine -> nouvel épisode
                            eps = self.training_parameters.min_epsilon + (self.training_parameters.max_epsilon - self.training_parameters.min_epsilon) * math.exp(-self.training_parameters.lbd * steps)
                            break
                    steps+=1
                    episodes_time.append(time.time() - t0)
                    pbar.update()
                    
                    
                #print("time boucle : ", time.perf_counter() - t1)
            
        self.env.close()
        pbar.close()

        self.parameters.Q = np.where(self.parameters.R == 1, self.parameters.Q, -math.inf)

        self.__save_Q_table(q_table_path)
        if training_time_path is not None:
            self.__save_training_time(episodes_time, training_time_path)

    def __save_Q_table(self, path):

        rows = []
        for state in self.parameters.states:
            rows.append(str(state))

        columns = []
        for action in self.parameters.actions:
            columns.append(str(action))

        df = pd.DataFrame(self.parameters.Q, index=rows, columns=columns)
        df.to_csv(path, sep=';')

    def __save_training_time(self, episodes_time, path):
        
        su = sum(episodes_time)
        mean_time = mean(episodes_time)
        med = median(episodes_time)
        deviation = std(episodes_time)

        rows = ['Time (s)','Mean (s)', 'Median (s)', 'std (s)']
        columns = ['training']

        l = [su, mean_time, med, deviation]

        a = np.asarray(l).transpose()

        df = pd.DataFrame(a, index=rows, columns=columns)
        df.to_csv(path, sep=';')