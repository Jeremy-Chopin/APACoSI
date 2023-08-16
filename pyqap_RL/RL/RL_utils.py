from ..metrics import dice_score, avg_dice_score
import random
import numpy as np
import itertools
from numpy.core.fromnumeric import sort
import pandas as pd
from numpy.core.fromnumeric import mean, std
from numpy.lib.function_base import median
import pandas as pd


    
def reward_from_dice_score(self, class_to_evaluate, region_to_evaluate, segmentation, gt):

    reward = 0
    if type(class_to_evaluate) is tuple:
        for i in range(0, len(class_to_evaluate)):
            reward += dice_score(class_to_evaluate[i], region_to_evaluate[i], segmentation, gt)
        reward = reward / len(class_to_evaluate)
    else:
            reward = dice_score(class_to_evaluate, region_to_evaluate, segmentation, gt)

    return reward

def get_random_actions_from_states_dict(dict, actual_state):

    possible_actions = dict[0][actual_state]
    chosen_action = random.choice(possible_actions)

    return chosen_action


def create_actions_states_Q_R_matrices(noeuds_model, s = 3):
    
    states, actions = __create_actions_states(noeuds_model, s)

    R = __create_R_matrix(states, actions)

    states.insert(0, "initial")

    Q = np.zeros((len(states) , len(actions)))

    return states, actions, Q, R

def __create_R_matrix(states, actions):
    
    R1 = np.zeros((1, len(actions)), dtype=np.uint8)

    for i in range(0, R1.shape[0]):
        for j in range(0, R1.shape[1]):
            if type(actions[j]) is tuple:
                R1[i][j] = 1

    R2 = np.zeros((len(states), len(actions)), dtype=np.uint8)

    for i in range(0, R2.shape[0]):
        for j in range(0, R2.shape[1]):
            if type(actions[j]) is not tuple:
                if actions[j] not in states[i]:
                    R2[i][j] = 1

    R = np.vstack((R1, R2))

    return R

def __create_actions_states(noeuds_model, s = 3):
    actions = []
    for iter in itertools.combinations(noeuds_model, s):
        actions.append(iter)

    for n in noeuds_model:
        actions.append(int(n))

    states = []

    for i in range(s, len(noeuds_model) + 1):
        for iter2 in itertools.combinations(noeuds_model, i):
            states.append(iter2)

    return states, actions

def create_states_actions_dict(R):
    actions_dict = {}

    for i in range(0, R.shape[0]):

        autorized_actions = np.where(R[i,:] == 1)
        actions_dict[i] = autorized_actions[0].tolist()
    
    return actions_dict

def create_new_states_matrix(actions, states, R):

    D = np.zeros(R.shape)
    x,y = np.where(R == 1)

    for i in range(0,len(x)):
        state = states[x[i]]
        action = actions[y[i]]

        if type(action) is tuple:
            new_state = action
        else:
            current_state_list = list(state)
            current_state_list.append(action)
            current_state_list = sort(current_state_list)
            current_state_tuple = tuple(current_state_list)
            new_state = current_state_tuple

        new_state_index = states.index(new_state)
        
        D[x[i]][y[i]] = new_state_index
    
    """
    D2 = np.zeros(R.shape, dtype=np.float32)
    for i in range(0, R.shape[0]):
        
        if i == 0:
            for j in range(0, R.shape[1]-15):
                if R[i][j] == 1:
                    state = states[i]
                    action = actions[j]
                    
                    if type(action) is tuple:
                        new_state = action
                    else:
                        print("Error")

                new_state_index = states.index(new_state)
        
                D2[i][j] = new_state_index
        else:
            for j in range(R.shape[1]-14, R.shape[1]):
                if R[i][j] == 1:
                    state = states[i]
                    action = actions[j]
                    
                    if type(action) is tuple:
                        print("Error")
                    else:
                        current_state_list = list(state)
                        current_state_list.append(action)
                        current_state_list = sort(current_state_list)
                        current_state_tuple = tuple(current_state_list)
                        new_state = current_state_tuple

                    new_state_index = states.index(new_state)
            
                    D2[i][j] = new_state_index
            
    """
                
    """if np.sum(np.where(D == D2, 0, 1) > 0):
        print("error")"""
            

    return D

def apply_matching(adjacency_matrix, segmentation):
	
	temp = np.zeros(segmentation.shape)

	x,y = np.where(adjacency_matrix == 1)

	for i in range(0, len(x)):
		temp = np.where(segmentation == x[i] + 1, y[i]+1, temp)

	return temp

def save_inference_time(episodes_time, path):
        su = sum(episodes_time)
        mean_time = mean(episodes_time)
        med = median(episodes_time)
        deviation = std(episodes_time)

        rows = ['Time (s)','Mean (s)', 'Median (s)', 'std (s)']
        columns = ['Testing']

        l = [su, mean_time, med, deviation]

        a = np.asarray(l).transpose()

        df = pd.DataFrame(a, index=rows, columns=columns)
        df.to_csv(path, sep=';')