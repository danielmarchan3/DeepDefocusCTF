
# DeepDefocusCTF

DeepDefocusCTF is a physics-constrained neural network designed to estimate defocus in both U and V directions, along with defocus angles, using the Power Spectral Density (PSD) of a micrograph in cryoEM.

## Repository Structure

```
DeepDefocusCTF/
│── prepare_training_dataset.py   # Script to prepare training dataset
│── train_model.py               # Script to train the model
│── predict.py             # Script for inference/predictions
│── utils/                      # Utility functions
│   ├── utils.py
│── models/                     # Neural network model definition
│   ├── deep_defocus_model.py
│── data_generator/              # Data loading and augmentation
│   ├── data_generator.py
│── trained_models/              # Directory to store trained models
│── README.md                   # Project description and usage
│── requirements.txt             # Dependencies
│── environment.yml              # Conda environment setup
│── .gitignore                   # Ignore unnecessary files
│── LICENSE                      # License file
```

## Installation

To set up the environment, use Conda:

```bash
conda env create -f environment.yml
conda activate deepdefocusctf
```

Or install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Dataset
Run the following command to prepare the dataset:
```bash
python prepareTrainingDataset.py --input <input_folder> --output <output_folder>
