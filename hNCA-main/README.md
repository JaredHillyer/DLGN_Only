# Hierarchical Neural Cellular Automata for Lightweight Microscopy Image Classification (hNCA)

## Overview
This project is designed for training and evaluating hNCA model using specified microscopy datasets and folds. It leverages Conda for environment management.

## Setup
### 1. Create Conda Environment
To set up the required environment, use the provided `env_hnca.yml` file:
```sh
conda env create -f env_hnca.yml
conda activate env_hnca
```

## Training
To train the model, run the following command:
```sh
python3 src/train.py --output #your_path# --train_set #your_dataset# --fold #your_fold# --mode train
```
Replace:
- `#your_path#` with the desired output directory.
- `#your_dataset#` with your dataset name.
- `#your_fold#` with a fold number (1-5).

## Evaluation
To evaluate the trained model, run:
```sh
python3 src/train.py --output #your_path# --train_set #your_dataset# --fold #your_fold# --mode eval
```
Ensure that `#your_fold#` is one of `[1, 2, 3, 4, 5]`.

## Additional Information
- Class-wise distribution of 6 datasets provided.
- Trained checkpoints for fold 1 of each dataset are provided in the folder results/.

## Contact
For any questions or issues, feel free to open an issue or reach out!
c95.yang@tum.de
