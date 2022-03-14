
import math as m
import sys
import os
import itertools
import seaborn as sn

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import GPy
from termcolor import colored

from .kernels import *
from .language import *
from .kernels_utils import *
from .training_utils import *
from .plotting_utils import *
from .search import preparekernel, decomposekernel
from .kernels_utils import KERNELS_FUNCTIONS, KERNELS_LENGTH

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')
PI = m.pi
_precision = tf.float64


def convert_tensor(X_train):
    X_train = tf.Variable(X_train, dtype=_precision)
    return X_train


class CustomModel(object):
    '''
        Custom model to do gaussian processes regression 
    '''

    def __init__(self, params, existing=None, X_train=None, use_noise=True):
        '''
            Initialize default class dic with parameters corresponding to kernels 
        inputs :
            params dic, containing parameters according to the kernels
            existing dic, containing parameters with converged values of previous training (in order to initialize optimization at previous minimum)
        outputs :
            None
        '''

        for attr in params.keys():
            pars = params[attr]
            for var in pars:
                if existing is None:
                    if var[:4] == "cp_s":
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=0.95, maxval=1.1))
                    elif var[:5] == "cp_x0":
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=5., maxval=float(len(X_train))))
                    else:
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=1e-2, maxval=20.))

                else:
                    if var in existing.keys():
                        self.__dict__[var] = tf.Variable(
                            existing[var], dtype=_precision)

                    elif var[:4] == "cp_s":
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=0.95, maxval=1.))
                    elif var[:4] == "cp_x0":
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=10., maxval=float(len(X_train))))
                    else:
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                                                                       dtype=_precision,
                                                                       shape=(
                                                                           1,),
                                                                       initializer=tf.random_uniform_initializer(minval=1e-2, maxval=20.))
        if use_noise:
            self.__dict__["noise"] = tf.compat.v1.get_variable("noise",
                                                               dtype=_precision,
                                                               shape=(1,),
                                                               initializer=tf.random_uniform_initializer(minval=1e-2, maxval=20.))
        else:
            self.__dict__["noise"] = tf.Variable(0., dtype=_precision)

    @property
    def initialisation_values(self):
        return dict({k: v for k, v in zip(list(vars(self).keys()), list(vars(self).values()))})

    @property
    def variables(self):
        return vars(self).values()

    @property
    def _variables(self):
        return vars(self)

    @property
    def _opti_variables(self):
        return list(vars(self).values())

    @property
    def _opti_variables_name(self):
        return vars(self).keys()

    # @tf.function
    def __call__(self, X_train, Y_train, kernels_name):
        """[summary]

        Args:
            X_train ([type]): [description]
            Y_train ([type]): [description]
            kernels_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        params = vars(self)
        return log_cholesky_l_test(X_train, Y_train, params, kernel=kernels_name)

    def view_parameters(self, kernels):
        """[summary]

        Args:
            kernels ([type]): [description]
        """
        list_vars = self.variables
        print("\n Parameters of  : {}".format(kernels))
        print("   var name               |               value")
        for name, value in zip(self._opti_variables_name, self.variables):
            print("   {}".format(str(name))+" "*int(23-int(len(str(name))))+"|" +
                  " "*int(23-int(len(str(value.numpy()))))+"{}".format(value.numpy()))

    def evaluate_posterior(self, X_train, Y_train, X_s, kernels_name, params):
        """[summary]

        Args:
            X_train ([type]): [description]
            Y_train ([type]): [description]
            X_s ([type]): [description]
            kernels_name ([type]): [description]
            params ([type]): [description]

        Returns:
            [type]: [description]
        """
        cov = self._get_cov(X_train, X_train, kernels_name, params)
        cov_ss = self._get_cov(X_s, X_s, kernels_name, params)
        cov_s = self._get_cov(X_train, X_s, kernels_name, params)
        mu, cov = self._compute_posterior(Y_train, cov, cov_s, cov_ss)
        return mu, cov

    def predict(self, X_train, Y_train, X_s, kernels_name):
        """[summary]

        Args:
            X_train ([type]): [description]
            Y_train ([type]): [description]
            X_s ([type]): [description]
            kernels_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        params = self._variables
        try:
            X_train = convert_tensor(X_train)
            Y_train = convert_tensor(Y_train)
            X_s = convert_tensor(X_s)
        except Exception as e:
            pass
        mu, cov = self.evaluate_posterior(
            X_train, Y_train, X_s, kernels_name, params)
        return mu, cov

    def _get_cov(self, X, Y, kernel, params):
        """[summary]

        Args:
            X_train (numpy array): training points
            Y_train (numpy array): training points
            kernel (list): list of kernels. Defaults to None.
            params ([type]): [description]

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        params_name = list(params.keys())
        cov = 0
        num = 0
        for op in kernel:
            op = op.replace(":DEC_SIG", "")
            op = op.replace(":INC_SIG", "")
            if op[0] == "+":
                method = KERNELS_FUNCTIONS[op[1:]]
                par = params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError(
                        "Method %s not implemented" % op[1:])
                cov += method(X, Y, [params[p] for p in par])
                num += KERNELS_LENGTH[op[1:]]
            elif op[0] == "*":
                method = KERNELS_FUNCTIONS[op[1:]]
                par = params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError(
                        "Method %s not implemented" % op[1:])
                cov = tf.math.multiply(cov, method(
                    X, Y, [params[p] for p in par]))
                num += KERNELS_LENGTH[op[1:]]
            elif op[0] == "C":
                kernel_list = op[3:-1].replace(" ", "").split(",")
                left_method = KERNELS_FUNCTIONS[kernel_list[0][2:-1]]
                par_name_left_method = params_name[num:num +
                                                   KERNELS_LENGTH[kernel_list[0][2:-1]]]
                num += KERNELS_LENGTH[kernel_list[0][2:-1]]
                right_method = KERNELS_FUNCTIONS[kernel_list[1][2:-1]]
                par_name_right_method = params_name[num:num +
                                                    KERNELS_LENGTH[kernel_list[1][2:-1]]]
                num += KERNELS_LENGTH[kernel_list[1][2:-1]]
                par_name_sigmoid, num = params_name[num:num+2], num+2
                cov += CP(X, Y, [params[p] for p in par_name_sigmoid], left_method, right_method, [
                          params[p] for p in par_name_left_method], [params[p] for p in par_name_right_method])
        return cov

    def compute_BIC(self, X_train, Y_train, kernels_name):
        """ Evaluate the Bayesian information ciriterion

        Args:
            X_train (numpy array): training points
            Y_train (numpy array): training points
            kernels_name (list): list of kernels
        Returns:
            [int]: Bayesian information criterion
        """
        params = self._variables
        n = tf.Variable(X_train.shape[0], dtype=_precision)
        k = tf.Variable(len(params), dtype=_precision)
        try:
            ll = log_cholesky_l_test(
                X_train, Y_train, params, kernel=kernels_name)
        except Exception:
            pass
        return -ll - 0.5*k*tf.math.log(n)

    def plot(self, mu, cov, X_train, Y_train, X_s, kernel_name=None):
        """ Plot the predicted points with the confidence interval

        Args:
            mu (tf tensor): predicted mean
            cov (tf tensor): predicted covariance
            X_train (numpy array): training points
            Y_train (numpy array): training points
            kernels_name (list): list of kernels. Defaults to None.
            X_s (tf tensor): predicted points 
        """
        Y_train, X_train, X_s = Y_train, X_train, X_s
        mean, stdp, stdi = get_values(
            mu.numpy().reshape(-1,), cov.numpy(), nb_samples=100)
        if kernel_name is not None and kernel_name[:2] != "CP":
            plt.title("kernel : "+''.join(kernel_name)[1:])
        elif kernel_name is not None:
            plt.title("kernel : C"+''.join(kernel_name)[:])
        plot_gs_pretty(Y_train, np.array(mean), X_train,
                       X_s, np.array(stdp), np.array(stdi))
        plt.show()

    def _compute_posterior(self, y, cov, cov_s, cov_ss):
        """ Compute posterior mean and covariance 

        Args:
            y ([type]): [description]
            cov ([type]): [description]
            cov_s ([type]): [description]
            cov_ss ([type]): [description]

        Returns:
            [type]: [description]
        """
        params = self._variables
        mu = tf.matmul(tf.matmul(tf.transpose(cov_s), tf.linalg.inv(
            cov+params["noise"]*tf.eye(cov.shape[0], dtype=_precision))), y)
        cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s), tf.linalg.inv(
            cov+params["noise"]*tf.eye(cov.shape[0], dtype=_precision))), cov_s)
        return mu, cov

    def split_params(self, kernel_list):
        """[summary]

        Args:
            kernel_list ([type]): [description]

        Returns:
            [type]: [description]
        """
        params = list(self._opti_variables_name)
        list_params = []
        pos = 0
        for element in kernel_list:
            if element[1] == "P" and element[:2] != "CP":
                list_params.append(params[pos:pos+3])
                pos += 3
            elif element[:2] == "CP":
                chgs_p = remove_useless_term_changepoint(element)
                for kernels in chgs_p:
                    if kernels[1] == "P":
                        list_params.append(params[pos:pos+3])
                        pos += 3
                    else:
                        list_params.append(params[pos:pos+2])
                        pos += 2
                list_params.append(params[pos:pos+2])
                pos += 2
            else:
                list_params.append(params[pos:pos+2])
                pos += 2
        return list_params

    def describe(self, kernel_list):
        """[summary]

        Args:
            kernel_list ([type]): [description]
        """
        list_params = self.split_params(kernel_list)
        params_dic = self._variables
        loop_counter = 0
        splitted, pos = devellopement(kernel_list)
        summary = "The signal has {} componants :\n".format(len(splitted))
        for element in splitted:
            summary = comment(
                summary, element, pos[loop_counter], params_dic, list_params) + "\n"
            loop_counter += 1
        summary = summary + \
            "\t It also has a noise component of {:.1f} .".format(
                self._variables["noise"].numpy()[0])
        print(colored('[DESCRIPTION]', 'blue'), summary)

    def decompose(self, kernel_list, X_train, Y_train, X_s):
        """[summary]

        Args:
            kernel_list ([type]): [description]
            X_train ([type]): [description]
            Y_train ([type]): [description]
            X_s ([type]): [description]
        """
        list_params = self.split_params(kernel_list)
        splitted, pos = devellopement(kernel_list)
        params_dic = self._variables
        loop_counter = 0
        cov = 0
        for element in splitted:
            kernels = decomposekernel(element)
            if len(kernels) < 1:
                kernels = preparekernel(element)
            list_of_dic = [list_params[position]
                           for position in pos[loop_counter]]
            merged = list(itertools.chain(*list_of_dic))
            dictionary = dict(zip(merged, [params_dic[one] for one in merged]))
            dictionary.update({"noise": self.__dict__["noise"]})
            decomp_model = CustomModel(params=kernels, existing=dictionary)
            mu, cov = decomp_model.predict(X_train, Y_train, X_s, element)
            plt.title("kernel :"+''.join(element)[1:])
            decomp_model.plot(mu, cov, X_train, Y_train, X_s, kernel_name=None)
            plt.show(block=True)
            plt.close()
            loop_counter += 1
            del decomp_model
