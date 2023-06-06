# Neural_Network_Project
## Introduction
This is our implementation of classical RNN networks,
including Vanilla, GRU and LSTM models, versus the novel architecture proposed by
Michael Rotman and Lior Wolf named Shuffling RNN.

# Data
## Classification Dataset
Our dataset is the IMDB datasaet.
IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This dataset is provided with a set of 50,000 highly polar movie reviews.
## Regression Dataset
The dataset used in the study consists of a multivariate time series related to wheather
conditions. Specifically, the time series consisted of four variables: humidity, pressure,
temperature, and wind speed.

# Libraries
* Pythorch
* Scikit Learn
* Pandas
* Matplotlib
* Numpy
* Torchtext
# Usage

Our project is divided into 2 differents part, the simple Python project and the Jupiter Notebook project.
## Python Project
Before to start install all the libraries listed before.

To start the program you have to be in the one of the 2 folders in the project, tiping `cd ./Classification` or `cd ./Regression`.
In the folder simply type `python main.py` or `python main.py`. 

### Arguments
The main functions take various arguments. To use test the training of one neural network `-t`, to reload a saved neural network `-r` and to train all the neural networks to plot our experiments `-i`.

#### Test Aguments
If type `-t` you can use more arguments:

* `-m` for the model you want to train, followed by the initial letter of the possible models: `rlsg`
* `-s` to save the model
* `-l` to select the learning rate
* `-p` to plot the loss
* `-e` to select the number of epochs
* `-b` to select the batch size
* `-d` to choose the directory of the dataset
* `-h` to select the hidden size

##Jupiter Notebook Project

To start the program you just have to have Jupyter Notebook installed on your system. If you don't have Jupyter Notebook installed or use sites like `Google Colab`. Then you have you execute the `Notebook_Neural_Network_Project.ipynb` file.

### Jupiter Notebook Structure

At the start of the Notebook there is an introduction about the method we used and the imported libraries.

The Notebook is divided in 2 parts, the classification and the regression one.
Every of these 2 sections is divided in sections that are the following:
* Utils functions we used
* The RNNs used for the test and the implementation of the LSTM
* The functions to run the tests with paramiters
* The tests

# Results
The results of our experiments are saved in the *Results* folder.
