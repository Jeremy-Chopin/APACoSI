# APACoSI

This project is related to the project APACoSI, and the code in this repository is related to this conference [paper](https://link.springer.com/chapter/10.1007/978-3-031-09037-0_15).

This conference paper described a method using structural knowledges to improve a semantic segmentation provided by a neural network.

## Methods

```python
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
proposal_matching, proposal_score = get_many_to_one_matching(...
        
proposal_image = create_images_from_ids(labelled_image, proposal_matching)
```

## Tutorial

An example on 2d synthetic dataset is available.

![Alt text](repository/proposal_example.png?raw=true "Example on a synthethic image.")

## License
[MIT](https://choosealicense.com/licenses/mit/)
## Citation

Please, cite this paper if you are using our algorithm.

@InProceedings{10.1007/978-3-031-09037-0_15,
author="Chopin, J{\'e}r{\'e}my and Fasquel, Jean-Baptiste and Mouch{\`e}re, Harold and Dahyot, Rozenn and Bloch, Isabelle",
editor="El Yacoubi, Moun{\^i}m and Granger, Eric and Yuen, Pong Chi and Pal, Umapada and Vincent, Nicole",
title="Improving Semantic Segmentation with Graph-Based Structural Knowledge",
booktitle="Pattern Recognition and Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="173--184",
isbn="978-3-031-09037-0"
}
