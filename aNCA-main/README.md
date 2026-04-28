# Attention Pooling Enhances NCA-based Classification of Microscopy Images (aNCA)

This repository provides code for training and evaluating the **aNCA** model introduced in  
[**Attention Pooling Enhances NCA-based Classification of Microscopy Images**](https://arxiv.org/abs/2508.12324).

aNCA extends Neural Cellular Automata (NCA) with an attention pooling mechanism, achieving competitive results on multiple microscopy datasets while maintaining a lightweight architecture.


## Setup
### 1. Create Conda Environment
To set up the required environment, use the provided `env_nca.yml` file:
```sh
conda env create -f env_nca.yml
conda activate env_nca
```

## Training
To train the model, run the following command:
```sh
python3 src/train.py --mode train --predict aNCA --output #your_path# --train_set #your_dataset# --fold #your_fold# 
```
Replace:
- `#your_path#` with the desired output directory.
- `#your_dataset#` with your dataset name.
- `#your_fold#` with a fold number (1-5).

## Evaluation
To evaluate the trained model, run:
```sh
python3 src/train.py --mode eval --predict aNCA --output #your_path# --train_set #your_dataset# --fold #your_fold# 
```
Ensure that `#your_fold#` is one of `[1, 2, 3, 4, 5]`.

To get a summary of the evaluation archive "res.pkl", run:
```sh
python3 res.py 
```

## Contact
For any questions or issues, feel free to open an issue or reach out!
c95.yang@tum.de
