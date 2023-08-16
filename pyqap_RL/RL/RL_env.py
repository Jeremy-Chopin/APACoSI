from time import time
import gym
import numpy as np
from gym import spaces

from ..metrics import dice_score, avg_dice_score
from ..QAP.QAP import reward_from_dice_score, find_best_matching
import time



class GymQapEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, states, actions, D):
        super(GymQapEnv, self).__init__()
        
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Discrete(len(states))

        print("There is {} states".format(len(states)))
        print("There is {} actions".format(len(actions)))

        self.actual_state = 0
        self.action_mapping = actions
        self.states = states
        self.precedent_action = []
        self.D = D

    def render(self):
        print(self.actual_assignement_matrix)

    def step(self, action, inference=False):
        done = False

        nb_states = len(self.states)
        nb_classes_to_match = len(self.states[nb_states-1])

        # label associé à l'action en cours
        class_to_match = self.action_mapping[action]

        if action in self.precedent_action:
            reward = 0
        else:
            
            self.actual_assignement_matrix, regions_matched = find_best_matching(class_to_match, self.actual_assignement_matrix, self.K)
            self.precedent_action.append(action)
            
            class_to_match = np.array(class_to_match)
            regions_matched = np.array(regions_matched)
            reward = reward_from_dice_score(class_to_match, regions_matched, self.segmentation, self.groundtruth)

        # Condition d'arret
        if reward <= 0.1:
            if inference is False:  
                done = True
                reward = -1
        else:
            reward +=1
        
        if np.sum(self.actual_assignement_matrix) == nb_classes_to_match:
            done = True
            #reward = 8

        self.actual_state = int(self.D[self.actual_state][action])
        
        return self.actual_state, reward, done, {}

    def reset(self, segmentation, groundtruth, K):

        self.segmentation = segmentation
        self.groundtruth = groundtruth
        self.K = K

        nb_class = int(np.max(groundtruth))
        nb_regions = int(np.max(segmentation))

        self.actual_assignement_matrix = np.zeros((nb_regions, nb_class))

        self.actual_state = 0
        self.precedent_action = []

        return self.actual_state
    
    def get_adjacency_matrix(self):
        return self.actual_assignement_matrix