# DeepTherm
This is the repository for the paper "DeepTherm: A Unified Deep Learning Approach to ‎Thermochemistry Prediction for Gas-phase Molecules‎".

DeepTherm is a deep learning framework designed to predict thermochemical properties, such as enthalpy of formation, entropy, and heat capacity for diverse molecular species. The project integrates directed message-passing networks and global attention mechanisms to capture both local and long-range dependencies in molecular graphs.

The datasets are provided along with the paper.

The code was built based on [DMPNN](https://github.com/chemprop/chemprop). Thanks a lot for their code sharing!

## Dependencies

+ cuda >= 8.0
+ cuDNN
+ RDKit
+ torch >= 1.2.0

Tips: Using code `conda install -c rdkit rdkit` can help you install package RDKit quickly.

## Directory Structure

### data                  
This directory contains the code for data processing and preprocessing. It also includes the scripts for data splitting, molecular feature extraction, and classification.

### features
This directory contains the code for atomic and bond feature extraction, as well as the generation of additional molecular descriptors.

### model
This directory contains the code for model definitions.

### train
This directory contains the code for model training, evaluation, and cross-validation.


## Data Analysis
Place the SMILES data files in the data/ directory.

Ensure data follows the format:

`SMILES,Hf(298K),S(298K),C300,C400,C500,C600,C800,C1000,C1500`

`COO,-30.08,67.3,14.34,16.8,19.09,21.07,24.19,26.52,30.25`

Run scripts:
`python data/scaffold.py --input <data_path> --output data/processed_data.csv`

## Feature Extraction
Extract initial features using the following command:

`python features/featurization.py --input <data_path> --output data/features.csv`

Initial features include:

Atom type, formal charge, hybridization, aromaticity, and more.

Bond type, conjugation, in-ring status, and additional bond features.

Generate molecular descriptors using the following command:

`python features/features_generators.py --input data/<data_path> --output data/features.csv`


## Training
To train a model, run:

`python train.py --data_path <path> --dataset_type <type> --num_folds <number_of_k-fold_cross_validation> --gpu <GPU_number> --epochs <epoch_number>`

where <path> is the path to a CSV file containing a dataset, <type> is either "classification" or "regression" depending on the type of the dataset.

Key hyperparameters include:
Learning Rate
Batch Size
Number of ‎Message-Passing Steps
Hidden Size
Number of Fully ‎Connected Layers
Dropout Rate
Weight Decay
Optimizer
Aggregation ‎Normalization
Graph Pooling Method
Attention Heads
Early Stopping Patience
Activation Function


## Predicting

`python predict.py --data_path <path> --checkpoint_dir <dir>`

where <dir> is the directory where the model checkpoint(s) are saved, and <path> is the path of SMILES dataset



### Acknowledgement 

This code was developed at the King Abdullah University of Science and Technology (KAUST).


