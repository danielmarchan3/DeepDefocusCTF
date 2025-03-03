
# DeepDefocusCTF

DeepDefocusCTF is a physics-constrained neural network designed to estimate defocus in both U and V directions, along with defocus angles, using the Power Spectral Density (PSD) of a micrograph in cryoEM.

## Repository Structure

```
DeepDefocusCTF/
│── prepareTrainingDataset.py   # Script to prepare training dataset
│── trainModel.py               # Script to train the model
│── predictModel.py             # Script for inference/predictions
│── Utils/                      # Utility functions
│   ├── utils.py
│── Models/                     # Neural network model definition
│   ├── DeepDefocusModel.py
│── DataGenerator/              # Data loading and augmentation
│   ├── dataGenerator.py
│── TrainedModels/              # Directory to store trained models
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
