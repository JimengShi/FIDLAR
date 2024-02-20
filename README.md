# FIDLAR
This repository contains the source code and datasets for "FIDLAR: Forecast-Informed Deep Learning Architecture for Flood Mitigation".

## Repository description
- `data` folder includes data sets used
- `model` folder includes models used
- `training_mlp` folder includes training programs for all datasets. 
- `postprocess` folder includes the programs for experiment results, visualization, and hyperparameter tuning.
- `saved_models_mlp` folder saves all the trained models at the best checkpoint.
- `saved_models_hyper` folder saves all the trained models while doing hyperparameter tuning for a major hyperparameter `number of frozen layers`.

## Requirements
```bash
conda create -n env_name python=3.8
conda activate env_name
pip3 install -r requirements.txt
```

## Running
- Download the entire repository and install the required packages (see requirements above).
- For training, go to the `training_mlp` folder and run cells in the `ipynb` files under each dataset.
- For testing and experiment analysis, go to the `postprocess` folder and run cells in the `ipynb` files.
