# APACoSI

This project is related to the project APACoSI, and the code in this repository is related to a journal paper and a conference paper.

This journal paper ([Model-based inexact graph matching on top of DNNs for semantic scene understanding, CVIU 2023](https://www.sciencedirect.com/science/article/abs/pii/S1077314223001248)) described a method using structural knowledges to improve a semantic segmentation provided by a neural network.

The conference paper ([QAP Optimisation with Reinforcement Learning for Faster Graph Matching in Sequential Semantic Image Analysis, ICPRAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-09037-0_5)) proposed an optimized version of the method described in the journal paper using reinforcement learning.

## Methods without Reinforcement Learning

```python
from pyqap.Nodes import MaxDistanceSpecifier, CnnProbabilitiesSpecifier
from pyqap.Edges import RelativePositionSpecifier
from pyqap.matching import get_one_to_one_matching, get_many_to_one_matching

# Paths and Images
pr_mask = np.load(os.path.join('test_cnn_output.npy'))
image_cnn = np.argmax(pr_mask, axis=2)

gt_mask, affine = load_file(os.path.join('test_gt.png'))
intensity_input, _ = load_file(os.path.join('test_rgb.png'))

KNOWLEDGES_DIR = os.path.join('knowledges')

# Parameters
params = {}
params['alpha'] = 0.5
params['Cs'] = get_diagonal_size(image_cnn)
params['weigthed'] = True
params['lbd'] = 0.5
params['min_max_coef'] = 0.5

nb_classes = 3

max_node_matching = 3
max_node_refinement = math.inf

# Attributes
nodes_specifier = [
    CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier(),
    MaxDistanceSpecifier.MaxDistandeSpecifier()
]

node_knowledge = load_knowledge(KNOWLEDGES_DIR, 'nodes', nodes_specifier)

edges_specifier = [
    RelativePositionSpecifier.RelativePositionSpecifier()
]

edge_knowledge = load_knowledge(KNOWLEDGES_DIR, 'edges', edges_specifier)

# Pre-processing
dims = len(image_cnn.shape)
pr_mask = pr_mask[:,:, 1:nb_classes + 1]
    
# initial matching
best_matching, best_score, labelled_image, regions = get_one_to_one_matching(...)

matching_image = create_images_from_ids(labelled_image, best_matching)

# refinement
proposal_matching, proposal_score = get_many_to_one_matching(...)
        
proposal_image = create_images_from_ids(labelled_image, proposal_matching)
```

## Methods with Reinforcement Learning

The following python code illustrate how to import and use the different functionalities of our RL-based algorithm (it is from the file main_RL.py).

```python
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
```

## Tutorials

### Synthetic dataset
An example for processing graph matching on 2d synthetic dataset is available (the explanation of this synthetic dataset is find in this conference [paper](https://link.springer.com/chapter/10.1007/978-3-031-09037-0_5)).

![Alt text](repository/proposal_example.png?raw=true "Example on a synthethic image.")

To execute this tutorial use the following command :
```
python main_no_RL.py
```

### FASSEG dataset

An example for processing graph matching using our RL-baseg algorithm on 2d FASSEG dataset is available.

To execute this tutorial use the following command :

```
python main_RL.py
```

### IBSR dataset

An example for processing graph matching using our RL-baseg algorithm on 3d IBSR dataset is available but the data are too heavy to be posted on this repository (contact me and I'll share the data with you).

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Citations

Please, cite this paper if you are using our algorithm.

```
@article{CHOPIN2023103744,
title = {Model-based inexact graph matching on top of DNNs for semantic scene understanding},
journal = {Computer Vision and Image Understanding},
volume = {235},
pages = {103744},
year = {2023},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2023.103744},
url = {https://www.sciencedirect.com/science/article/pii/S1077314223001248},
author = {Jeremy Chopin and Jean-Baptiste Fasquel and Harold Mouchère and Rozenn Dahyot and Isabelle Bloch}
}
```

Please, cite this paper if you are using our optimized algorithm using reinforcement learning.

```
@InProceedings{10.1007/978-3-031-09037-0_5,
author="Chopin, J{\'e}r{\'e}my and Fasquel, Jean-Baptiste and Mouch{\`e}re, Harold and Dahyot, Rozennand Bloch, Isabelle",
editor="El Yacoubi, Moun{\^i}m and Granger, Eric and Yuen, Pong Chi and Pal, Umapada and Vincent, Nicole",
title="QAP Optimisation with Reinforcement Learning for Faster Graph Matching in Sequential Semantic Image Analysis",
booktitle="Pattern Recognition and Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="47--58"
}
```
