# APACoSI

This project is related to the project APACoSI, and the code in this repository is related to this conference [paper](https://link.springer.com/chapter/10.1007/978-3-031-09037-0_15).

This conference paper described a method using structural knowledges to improve a semantic segmentation provided by a neural network.

## Methods

The structural informations are embedded on vertices and edges of the graph. Those structural informations are evaluated through the "Nodes" and "Edges" functions. 

```python
nodes_specifier = [
    CnnProbabilitiesSpecifier.CnnProbabilitiesSpecifier(),
    MaxDistanceSpecifier.MaxDistandeSpecifier()
]

edges_specifier = [
    RelativePositionSpecifier.RelativePositionSpecifier()
]
```

The K matrix is the matrix that embeddes the dissimilarities between the two graphs and dissimilarities functions are used to evaluate the difference of attributes on the graphs.

Each structural informations is related to a weigth parameter.

```python
kv_constructor = KvConstructor(nodes_specifier, [0.5,0.5], node_knowledge)
        
kv = kv_constructor.construct_Kv(pr_mask, labelled_image, regions, matching, params)

ke_constructor = KeConstructor(edges_specifier, [1], edge_knowledge)

ke = ke_constructor.construct_Ke(pr_mask, labelled_image, regions, matching, params)
```

The inexact graph matching operation guided by the K matrix is decribed in the tutorial.

## Tutorial

An example on 2d synthetic dataset is available.

![Alt text](repository/proposal_example.png?raw=true "Example on a synthethic image.")

## License
[MIT](https://choosealicense.com/licenses/mit/)
