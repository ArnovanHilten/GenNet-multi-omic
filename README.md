# GenNet-multi-omic


Uses the GenNet framework to create interpretable neural networks for multi-omics data. This reposisitory contains the code to recreate multi-omic neural networks for gene-expression and methylation input. 



<img src="https://github.com/ArnovanHilten/GenNet-multi-omic/blob/main/images/Multi-omics%2C%20figure%201.svg">

Above a schematic overview of the neural network architectures. In the ME network (a), DNA methylation data (CpGs) are grouped and connected using gene annotations. The resulting 10,404 gene nodes are directly connected to the output node. Combining the ME network and the the GE network (b) for gene expression, results in the ME+GE network (c). In the ME+GE network each gene has a node per omic and a combined gene representation. Design (d) adds a covariate input to the combined gene representation for each gene. This allows the ME+GE network to model gene-specific effects for the covariate.


A schematic overview of the pathway network:

<img src="https://github.com/ArnovanHilten/GenNet-multi-omic/blob/main/images/Networks_Sup_%20(1).png">


## Get started
This repository uses basic functions and the LocallyDirected layer [here](https://github.com/ArnovanHilten/GenNet/blob/master/GenNet_utils/LocallyDirectedConnected_tf2.py)
<a name="how"/> the GenNet repository (added as a submodule). To get started check the [GenNet repository](https://github.com/ArnovanHilten/GenNet/#2-getting-started)

In the future multi-omics, or rather multi-input networks will be available in the GenNet framwork.

For questions, problems or comments please email arnovanhilten@gmail.com or open an issue.

##




