from os.path import join

from pyqap_RL.QAP.KnowledgesNodesConstructor import KnowledgesNodesConstructor
from pyqap_RL.QAP.Nodes import CnnProbabilitiesSpecifier, MaxDistanceSpecifier
from pyqap_RL.QAP.KnowledgeEdgesConstructor import KnowledgesEdgesConstructor
from pyqap_RL.QAP.Edges import MinMaxEdtDistanceSpecifier
from pyqap_RL.RL import training_parameters

from pyqap_RL.matching import create_prior_knowledge, create_K_matrices, training, inference

# Data information
nb_classes = 8

# Paths
data_path = join('tutorial_RL', 'data')
experiments_path = join('tutorial_RL', 'experiments')

# QAP_Parameters
nodes_specifier = [
    CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier()
]

node_knowledge_constructor = KnowledgesNodesConstructor(specifier_list=nodes_specifier)

edges_specifier = [
    MinMaxEdtDistanceSpecifier.MinMaxEdtDistandeSpecifier()
]

edges_knowledge_constructor = KnowledgesEdgesConstructor(specifier_list=edges_specifier)

alpha = 0.5

nodes_parameters = {
    'weigthed' : True
}

edges_parameters = {}      
edges_parameters['weigthed'] = True
edges_parameters['lbd'] = 0.5
edges_parameters['min_max_coef'] = 0.5

modes = ['test', 'train']

# Function to evaluate the structural information of the model using label from the training data
create_prior_knowledge(nb_classes, data_path, experiments_path, node_knowledge_constructor, edges_knowledge_constructor, nodes_parameters)

# Function to estimate the dissimiarity matrix (K) between the graph model and graphs of the processed images
create_K_matrices(nb_classes, alpha, modes,data_path, experiments_path, nodes_specifier, edges_specifier, edges_parameters)

# Training parameters

seed_sizes = [2, 3]

tr_param = training_parameters.TrainingParameters(
    learning_rate=0.1,
    discount_factor=0.95,
    min_epsilon=0.05,
    max_epsilon=1,
    lbd=0.01
)

# Training the Q function to learn the "optimal" sequence for the matching using the training data
training(nb_classes, seed_sizes, data_path, experiments_path, tr_param)

# Inference using the "optimal" sequence on the test data
inference(nb_classes, seed_sizes, data_path, experiments_path)