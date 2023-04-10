import numpy as np
import pickle
import os
import py_classes.preprocess as prep

current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file)
framework_directory = os.path.abspath(os.path.join(parent_directory, '../../Framework/'))

import utils
from neuralNetwork import NeuralNetwork
from standardScaler import StandardScaler


x_train, y_train, x_validation, y_validation, scaler = prep.loadXY()

print(np.sum(y_train, axis=0))

NN = NeuralNetwork(x_train, y_train, 256, 0.01, 4000, 256, 128)
NN.train()
NN.save("./data/model/NN.pkl")