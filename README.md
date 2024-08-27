# Python Paillier implementation

Computer Science Studio 2 Assignment on applying Paillier encryption for neural networks to use encrypted data for training and testing

## Table of Contents

- [Installation](#installation)
- [Command-Line Arguments](#cmd)
- [Usage](#usage)

## Installation

Instructions on how to install and set up the project.

```bash
# Install python (linux)
sudo apt-get install python3

# Install requiremnets
pip install requirements.txt

```
Install python on website or microsoft store if on windows
## Command-Line Arguments

This script accepts several command-line arguments to configure its behavior. Below is a description of the available arguments and examples of how to use them. If no args are provided, it will run MLP with encrypted data.

### Available Arguments

- `--state`: An integer argument that specifies the whether data is encrypted or not. Put 1 for encrypted and 0 for unencrypted data. This argument is optional.
- `--model`: An integer argument that specifies the whether data is encrypted or not. Put 1 for Recurrent Neural Network and 0 for multi-layer perceptron. This argument is optional.


## Installation

Instructions on how to install and set up the project.

```bash
# Install python (linux)
sudo apt-get install python3

# Install requiremnets
pip install requirements.txt
```
Install python on website or microsoft store if on windows

## Usage
# run the main file to perform neural network (MLP/RNN) training with unencrypted/encrypted data
```bash
python3 main.py --state <state_value> --model <model_value>
```

