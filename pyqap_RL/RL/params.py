class Params:

    def __init__(self, actions, states, actions_dict, Q, R, D, nb_episodes_per_image : int = 1):
        self.actions = actions
        self.states = states
        self.actions_dict = actions_dict,
        self.Q = Q
        self.D = D
        self.R = R
        self.nb_episodes_per_image = nb_episodes_per_image
    
    def print(self):
        print("ok")