class Nodes_data():
    
    def __init__(self, ids, max_label_id, probability_vector, is_confusion, is_classified):
        self.ids = ids
        self.max_label_id = max_label_id
        self.probability_vector = probability_vector
        self.is_confusion = is_confusion
        self.is_classified = is_classified