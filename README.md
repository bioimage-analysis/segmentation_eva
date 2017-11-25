Segmentation Eva - Alex Dunn's lab
==================================

![Model](doc/g1.png?raw=true "Model Architecture")

## Setup
1- Download [Anaconda](https://www.anaconda.com/download/), a free installer that includes Python and all the common scientific packages.

2- Create a conda environment with an Ipython kernel:

```
 conda create --name name_env python=3 ipykernel
```

3- Activate your conda environment:

```
source activate name_env
```
4- Install files in the requirements.txt :

```
pip install -r requirements.txt
```

## Usage

Either use the Jupyter notebook:

cd to the notebook directory and lunch jupyter notebook:

```
jupyter notebook
```
or use the batch.py file:  
```
python batch.py --path_source PATH/TO/FILE -drug ctrl -drug Y2 --hard_sup true

```
## Contact
Cedric Espenel  
E-mail: espenel@stanford.edu
