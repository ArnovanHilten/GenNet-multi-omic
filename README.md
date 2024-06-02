# GenNet-multi-omic

This repository contains the code to reproduce the experiments in: [Phenotype prediction using biologically interpretable neural networks on multi-cohort multi-omics data](https://www.biorxiv.org/content/10.1101/2023.04.16.537073v1.full.pdf)

> [!NOTE]
> This reposisitory contains the code to recreate multi-omic neural networks for gene-expression and methylation input using the GenNet framework. In time, the extra functionalities introduced in this repository will also be available in the GenNet framework. 

<hr>


<img src="https://github.com/ArnovanHilten/GenNet-multi-omic/blob/main/images/Figure1.png">

Above a schematic overview of the neural network architectures. In the ME network (a), DNA methylation data (CpGs) are grouped and connected using gene annotations. The resulting 10,404 gene nodes are directly connected to the output node. Combining the ME network and the the GE network (b) for gene expression, results in the ME+GE network (c). In the ME+GE network each gene has a node per omic and a combined gene representation. Design (d) adds a covariate input to the combined gene representation for each gene. This allows the ME+GE network to model gene-specific effects for the covariate.


A schematic overview of the pathway network:

<img src="https://github.com/ArnovanHilten/GenNet-multi-omic/blob/main/images/Figure2.png">


## Get started
This repository uses basic functions and the LocallyDirected layer [here](https://github.com/ArnovanHilten/GenNet/blob/master/GenNet_utils/LocallyDirectedConnected_tf2.py)(added as a submodule). To get started check the [GenNet repository](https://github.com/ArnovanHilten/GenNet/#2-getting-started). In the future multi-omics, or rather multi-input networks will be available in the GenNet framwork. 


### Install the virtual envionment
This automatically installs the latest Tensorflow version for which GenNet has been tested. If you have an older version of CUDA install the appriopriate tensorflow-gpu by
`pip install tensorflow-gpu==1.13.1` (change 1.13.1 to your version).

**Navigate to the home folder and create a virtual environment**
```
cd ~
python3 -m venv env_GenNet
```

**Activate the environment**
```
source ~/env_GenNet/bin/activate
```

**Install the packages**
```
pip3 install --upgrade pip
pip3 install -r requirements_GenNet.txt

```
*Installation complete!, check the wiki to get started*


## Support

For questions, problems or comments please email arnovanhilten@gmail.com or open an issue.






