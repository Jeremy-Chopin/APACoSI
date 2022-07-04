import os
import math
import numpy as np

from matplotlib import pyplot as plt
from utils import load_knowledge, load_file, get_diagonal_size, create_images_from_ids

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
best_matching, best_score, labelled_image, regions = get_one_to_one_matching(
    nb_classes=nb_classes,
    params=params,
    image_cnn=image_cnn,
    pr_mask=pr_mask,
    node_knowledge=node_knowledge,
    edge_knowledge=edge_knowledge,
    nodes_specifier=nodes_specifier,
    edges_specifier=edges_specifier,
    nodes_specifier_weigths=[0.5,0.5],
    edges_specifier_weigths=[1]
)

matching_image = create_images_from_ids(labelled_image, best_matching)

# refinement
proposal_matching, proposal_score = get_many_to_one_matching(
    nb_classes=nb_classes,
    params=params,
    pr_mask=pr_mask,
    labelled_image=labelled_image,
    regions=regions,
    best_matching=best_matching,
    best_score=best_score,
    node_knowledge=node_knowledge,
    edge_knowledge=edge_knowledge,
    nodes_specifier=nodes_specifier,
    edges_specifier=edges_specifier,
    nodes_specifier_weigths=[0.5,0.5],
    edges_specifier_weigths=[1]
)
        
proposal_image = create_images_from_ids(labelled_image, proposal_matching)

# Results
plt.subplot(1,4,1); plt.title("Annotation"); plt.imshow(gt_mask)
plt.subplot(1,4,2); plt.title("CNN"); plt.imshow(image_cnn)
plt.subplot(1,4,3); plt.title("one-to-one"); plt.imshow(matching_image)
plt.subplot(1,4,4); plt.title("proposal"); plt.imshow(proposal_image)
plt.tight_layout()
plt.show()