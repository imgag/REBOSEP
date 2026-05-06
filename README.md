## **REBOSEP**: REgion BOundary Specific Expression Patterns
REBOSEP is a workflow designed for detecting gene expression patterns across tissue boundaries using spatial transcriptomics data. REBOSEP supports tissue boundary identification and reports gene expression as a function of boundary distance.  

## Resource
The gene expression profiles of the five boundary regions presented in our publication can be downloaded from [resource](https://github.com/imgag/REBOSEP_resource).

## Tool
### Installation
REBOSEP can be installed using pip:
```
git clone https://github.com/imgag/REBOSEP.git
conda create -y -n rebosep_env python=3.10
conda activate rebosep_env
cd REBOSEP
conda install -c conda-forge -c main -c anaconda --file requirements.txt
pip install .
```

### Example
An example workflow including a small dataset for testing purposes is included in the `example` directory. We recommend Jupyter Server to run the notebook.  
Note: the packages required for the `tutorial_clustering.ipynb` script are not included in the REBOSEP installation. They have to be installed separately.
