import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler
import sys
import time
import os
import ConfigSpace
from hyperopt import hp, STATUS_OK
from copy import deepcopy

class Simple_NN(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, layer3_size, dropout_rate):
        super(Simple_NN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(input_size, layer1_size),
            nn.BatchNorm1d(layer1_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.layer_2 = nn.Sequential(
            nn.Linear(layer1_size, layer2_size),
            nn.BatchNorm1d(layer2_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.layer_3 = nn.Sequential(
            nn.Linear(layer2_size, layer3_size),
            nn.BatchNorm1d(layer3_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate))
        self.regressor = nn.Linear(layer3_size, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        y = self.regressor(x)
        return y

class TuneNN(object):

    def __init__(self, data_dir='./', task='cifar10-valid', seed=None, bo_method='gp'):

        self.bo_method = bo_method
        self.task = task
        self.seed = seed
        # Load and process data
        # self.input_size = len(data.columns)
        # self.X_train, self.Y_train, self.X_valid, self.Y_valid =

    def objective_function(self, config, epochs=100):

        # minimise validation error
        layer1_size = config['layer1_size']
        layer2_size = config['layer2_size']
        layer3_size = config['layer3_size']
        dropout_rate = config['dropout_rate']

        net = Simple_NN(self.input_size, 2 ** layer1_size, 2 ** layer2_size, 2 ** layer3_size, dropout_rate)
        net.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=np.exp(config['lr']), weight_decay=np.exp(config['weight_decay']))
        num_epochs = config['num_epochs']

        # Training code

        # Validation code
        val_error = 1
        run_cost = 1
        return val_error, run_cost

    def eval(self, config, epochs=100):

        if self.bo_method == 'tpe':
            config_standard = deepcopy(config)
            for h in self.space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                    config_standard[h.name] = h.sequence[int(config[h.name])]
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    config_standard[h.name] = int(config[h.name])
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    config_standard[h.name] = int(config[h.name])

            y, c = self.objective_function(config_standard, epochs)
            return {
                'config': config,
                'loss': y,
                'cost': c,
                'status': STATUS_OK}

        elif self.bo_method == 'bohb':
            y, c = self.objective_function(config, epochs)
            return y, c

        elif self.bo_method == 'gp':
            config_standard = {}
            for j, h in enumerate(self.search_space):
                config_standard[h['name']] = config[j]

            y, c = self.objective_function(config_standard, epochs)
            return y, c

    def get_search_space(self):
        space = self.get_configuration_space()
        self.space = space
        if self.bo_method == 'tpe':
            search_space = {}
            for h in space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
                    search_space[h.name] = hp.quniform(h.name, 0, len(h.sequence) - 1, q=1)
                elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                    search_space[h.name] = hp.choice(h.name, h.choices)
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    search_space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    search_space[h.name] = hp.uniform(h.name, h.lower, h.upper)
        elif self.bo_method == 'gp':
            search_space = []
            for h in space.get_hyperparameters():
                if type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
                    type_name = 'categorical'
                    domain = h.choices
                elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
                    type_name = 'continuous'
                    domain = (h.lower, h.upper)
                elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
                    type_name = 'discrete'
                    domain = tuple(range(h.lower, h.upper))
                h_dict = {'name': h.name, 'type': type_name, 'domain': domain}
                search_space.append(h_dict)

        self.search_space = search_space
        return search_space

    @staticmethod
    def get_configuration_space():
        space = ConfigSpace.ConfigurationSpace()
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"lr", np.log(1e-4), np.log(5e-2)))
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"weight_decay", np.log(1e-5), np.log(1e-3)))
        space.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(f"dropout_rate", 0.0, 0.9))
        space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(f"num_epochs", 5, 30))
        space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(f"batch_size", 5, 10))
        for i in range(2):
            space.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter(f"layer{i+1}_size", 2, 11))

        return space

if __name__ == '__main__':
    from GPyOpt.experiment_design import initial_design
    from GPyOpt.core.task.space import Design_space

    problem = TuneNN()
    cs = problem.get_search_space()

    space = Design_space(cs, None)

    X = initial_design('random', space, 10)
