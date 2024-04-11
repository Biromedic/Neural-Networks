# Neural Network Hyperparameter Tuning Homework

## Introduction
This homework focuses on experimenting with various hyperparameters of a Two-Layer Multilayer Perceptron (MLP) to solve a regression problem. 
The goal is to understand how different hyperparameters, such as hidden layer sizes, learning rates, and others, affect the performance of the neural network.

## Dataset
- Datasets are given in the folder as one XOR dataset, and one Admission.csv file. 
## Requirements
List the libraries and tools required to run this project, for example:
- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Pandas

## Implementation Details

### Neural Network Architecture
The neural network consists of an input layer, one hidden layer, and an output layer. The activation function used between the input and hidden layer is the sigmoid function, providing a non-linear transformation of the inputs. 
For the output layer, the sigmoid function is again used to map the final values to a probability score between 0 and 1, suitable for binary classification tasks like the XOR problem.

- **Input Layer:** The size of the input layer corresponds to the number of features in the dataset.
- **Hidden Layer:** The number of neurons in the hidden layer is varied during the experiments (e.g., 10, 20, 40, 80, etc.) to study its effect on the model's performance.
- **Output Layer:** Comprises a single neuron as it is a binary classification problem.

### Hyperparameters Being Tuned
The primary hyperparameters tuned in this project are:
- **Hidden Layer Size:** The number of neurons in the hidden layer. Different configurations are tested to observe how increasing complexity affects learning.
- **Learning Rate:** The step size at each iteration while moving toward a minimum of a loss function. Values like 0.001, 0.01, and 0.1 are experimented with.
- **Number of Epochs and Batch Size:** These parameters control the learning process's duration and the amount of data passed through the network before updating the model parameters.

### Grid Search Strategy
A grid search strategy is employed to systematically work through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. The process involves:

1. Defining a set of values for each hyperparameter.
2. Training a model for every possible combination of hyperparameter values and evaluating its performance on a validation set.
3. Selecting the hyperparameter values that yield the best performance metrics (e.g., R^2 score, RMSE).
