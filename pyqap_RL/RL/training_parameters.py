class TrainingParameters:

    def __init__(self, learning_rate, discount_factor, min_epsilon, max_epsilon, lbd):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.lbd = lbd