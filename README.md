# Exploring the Impact of Batch Size on Model Fusion: Generalization, Curvature, and Optimization Dynamics

In this repository you will find the code and experiments used in our paper.

## Setup

Step 1. Create a virtual environment and activate it
`python -m venv .venv`

Step 2. If you want to use PyTorch with CUDA, install PyTorch by following the instructions from [here](https://pytorch.org/get-started/locally/)

Step 3. Install the Python requirements
`pip install -r requirements.txt`

Step 4. Install the repository as a package
`pip install -e .`

## Experiments

The code for running our experiments, along with the experiment configurations, can be found in `./Notebooks/experiments.ipynb`. To select which configuration to use, set `experiment_config` to the appropriate config variable, as specified in the notebook.

Additionally, we provide the code that we used to train the parent models. The script to train all models can be found in `./Experiments/run_all_training_scripts.py`.

Descriptions and visualizations of training runs can be found [here](https://wandb.ai/model-fusion/Model%20Fusion).

## Repository structure

./Experiments/ contains Python scripts for training parent models and applying LMC and PyHessian functions.

./model_fusion/ contains dataset and model descriptions, an implementation of model fusion, tools for LMC and PyHessian. For convenience, it is structured as a Python package.

./Notebooks/ contains the main experiment notebook
