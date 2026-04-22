## **REBOSEP**: REgion BOundary Specific Expression Patterns
REBOSEP is a workflow designed for detecting gene expression patterns across tissue boundaries using spatial transcriptomics data. REBOSEP supports tissue boundary identification and reports gene expression as a function of boundary distance.  
We have included a resource of gene expression patterns across five different brain-meninges boundaries in the developing mouse brain: dorsal pallium - meninges, midbrain - meninges, lower hindbrain - meninges, upper hindbrain - meninges, subpallium - meninges.

## Resource
The gene expression profiles of five boundary regions can be downloaded from [resource](https://github.com/imgag/REBOSEP_resource/resource).

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
