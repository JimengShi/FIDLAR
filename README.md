# FIDLAR
This repository contains the source code and datasets for "FIDLAR: Forecast-Informed Deep Learning Architecture for Flood Mitigation".

## Repository description
- `data` folder includes data sets used
- `baseline` folder includes baseline models used
- `model` folder includes our proposed models
- `loss` folder includes loss functions used
- `preprocess` folder includes data pre-processing
- `postprocess` folder includes the programs for experiment results, visualization, and ablation study
- `training_WaLeF_models` folder includes training programs for `Flood Evaluator` with all models
- `training_optimization_models` folder includes training programs for `Flood Manager` with frozen `Flood Evaluator`


## Requirements
```bash
conda create -n env_name python=3.8
conda activate env_name
pip3 install -r requirements.txt
```

## Running
- Download the entire repository and install the required packages (see requirements above).
- For training,
  - `Flood Evaluator`, go to the `training_WaLeF_models` folder and run cells in the `ipynb` files
  - `Flood Manager`, go to the `training_optimization_models` folder and run cells in the `ipynb` files
- For testing and experiment analysis, go to the `postprocess` folder and run cells in the `ipynb` files.
