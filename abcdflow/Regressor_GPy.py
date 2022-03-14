import sys
import os
import itertools
import math as m

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import GPy
from termcolor import colored

from .language import *
from .kernels_utils import *
from .training_utils import *
from .plotting_utils import *
from .search import preparekernel, decomposekernel
from .kernels_utils import KERNELS_FUNCTIONS


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.backend.set_floatx("float64")
PI = m.pi
_precision = tf.float64


class GPyWrapper:
    def __init__(self, model, kernels):
        self._model = model
        self._kernels_list = kernels

    def variables(self):
        return self._model.param_array

    def variables_names(self):
        return self._model.parameter_names()

    def view_parameters(self, kernels):
        print("\n Parameters of  : {}".format(kernels))
        print(self._model)

    def plot(
        self, mu=None, cov=None, X_train=None, Y_train=None, X_s=None, kernel_name=None
    ):
        self._model.plot()
        plt.show()
        return 0

    def split_params(self, kernel_list, params_values):
        list_params = []
        pos = 0
        for element in kernel_list:
            if element[1] == "P":
                list_params.append(params_values[pos: pos + 3])
                pos += 3
            elif element[1] == "L":
                list_params.append(params_values[pos: pos + 1])
                pos += 1
            else:
                list_params.append(params_values[pos: pos + 2])
                pos += 2
        return list_params

    def describe(self, kernel_list):
        splitted, pos = devellopement(kernel_list)
        loop_counter = 0
        variables_names = self.variables_names()
        variables = self.variables()
        list_params = self.split_params(kernel_list, variables)
        summary = "The signal has {} componants :\n".format(len(splitted))
        for element in splitted:
            summary = (
                comment_gpy(
                    summary, element, pos[loop_counter], variables_names, list_params
                )
                + "\n"
            )
            loop_counter += 1
        summary = summary + \
            "\t It also has a noise component of {:.1f} .".format(
                variables[-1])
        print(summary)

    def view_parameters(self, kernels):
        print(self._model)

    def decompose(self, kernel_list, X_train, Y_train, X_s):
        splitted, pos = devellopement(kernel_list)
        variables = self.variables()
        list_params = self.split_params(kernel_list, variables)
        loop_counter = 0
        counter = 0
        for element in splitted:
            kernels = preparekernel(element)
            list_of_dic = [list_params[position]
                           for position in pos[loop_counter]]
            params = [list_params[position] for position in pos[loop_counter]]
            loop_counter += 1
            try:
                k = self._gpy_kernels_from_names(element, params)
            except Exception as e:
                print(e)
            model = GPy.models.GPRegression(
                X_train, Y_train, k, normalizer=False)
            model.plot()
            loop_counter += 1

    def predict(self, X, Y, X_s, kernel):
        return self._model.predict(X_s)
