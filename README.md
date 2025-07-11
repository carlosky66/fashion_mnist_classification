# Fashion MNIST Classifier using aiforge

This project implements a neural network to classify images from the Fashion MNIST dataset using the TensorFlow Basic Classification tutorial, with the implementation built around a custom library named aiforge.

## Overview 
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories of clothing items. Each image is 28x28 pixels. The goal of this project is to train a model that can accurately classify these images into their respective categories.

This project leverages:

* The Hydra framework for flexible configuration management

* Custom training/evaluation logic provided by the `aiforge` library

* TensorFlow for the deep learning backend


## Prerequisites
Make sure the `aiforge` library is available in your environment. If not, you need to install it locally or from a repository.

### Installation of `aiforge`:

To install the `aiforge` package, follow the instructions in this link: [aiforge](https://github.com/carlosky66/aiforge)

## Installation

It's recommended to use a virtual environment to manage dependencies. You can use conda or pyenv, but we suggest using [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for faster dependency resolution.

### Create the environment

Using *conda*:
```
conda env create -f env.yml
```

Using *mamba*:
```
mamba env create -f env.yml
```

### Activate the environment
```
conda activate fashion-mnist
```

### Install required libraries
Make sure all required Python packages are installed:

```
pip install -r requirements.txt
```


## Usage

You can either train a new model or evaluate a previously trained one.

### Configuration
Before running any command, you must configure the YAML files located in the config/ directory. These files define hyperparameters, model architecture, the model saving path, output directories, and more.

Hydra will automatically handle these configurations during runtime.

### Training the model

To start training:
```
python3 main.py train
```
### Evaluating the model

To start evaluating:

To evaluate the model, execute the following command:
```
python3 main.py evaluate
```

## Project Structure
```
.
├── config/               # YAML config files for Hydra
├── models/               # Saved model checkpoints
├── .gitignore            # Gitignore of the project
├── env.yml               # Conda environment file
├── evaluate.py           # Evaluation logic
├── main.py               # Entry point
├── LICENSE               # MIT License
├── README.txt            # Project description and instructions
├── requirements.txt      # Python package dependencies
└── train.py              # Training logic
```

## Notes

* You can customize the training process by editing config/train.yaml or config/model.yaml.

    Hydra allows dynamic overrides via command line, e.g.:
    ```
    python3 main.py train model.hidden_units=256
    ```

* Make sure your working directory is set correctly when running the scripts so that Hydra can locate the config/ folder.