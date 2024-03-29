import pickle
import contextlib
import functools
import os
import time
from pprint import pprint
import logging
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math as m
import GPy
import pandas as pd
import seaborn as sn


from ..kernels.kernels import *
from ..kernels.kernels_utils import *
from ..regressors.Regressor_GPy import GPyWrapper

logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.backend.set_floatx("float64")


PI = m.pi
_jitter = 1e-7
_precision = tf.float64


def save_and_plot(func):
    """
    Decorator to plot the figure and pickle the model if user validate it.
    """

    def wrapper_func(*args, **kwargs):
        print(func, *args, **kwargs)
        model, kernels = func(*args, **kwargs)
        name = "./best_models/best_model"
        do_plot, save_model = args[-3], args[-2]
        X_train, Y_train, X_s = args[0], args[1], args[2]
        if do_plot:
            try:
                mu, cov = model.predict(X_train, Y_train, X_s, kernels)
                model.plot(mu, cov, X_train, Y_train, X_s, kernels)
            except:
                model.plot()
            plt.show()
        if save_model:
            with open(name, "wb") as f:
                pickle.dump(model, f)
            with open("./best_models/kernels", "wb") as f:
                pickle.dump(kernels, f)
        return model, kernels

    return wrapper_func


def print_trainning_steps(count, train_length, combinaison_element):
    """
        Print the avancing of the training ex, ==>..|
    inputs :
        count, int, actual training step
        train_length, int max training step
        combinaison_element, tuple containing the model's kernel
    outputs :
        None
    """
    sys.stdout.write(
        "\r"
        + "=" * int(count / train_length * 50)
        + ">"
        + "." * int((train_length - count) / train_length * 50)
        + "|"
        + " * model is {} ".format(combinaison_element)
    )
    sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def define_boundaries(model, X_train):
    bnds = []
    for param in model._opti_variables_name:
        if param == "cp_x0":
            bnds.append([5, len(X_train)])
        elif param == "cp_s":
            bnds.append([0.99, 1])
        elif param == "periodic_p" or param == "periodic_l" or param == "squaredexp_l":
            bnds.append([1e-8, len(X_train)])
        else:
            bnds.append([1e-8, None])
    return bnds


def update_current_best_model(
    BEST_MODELS, model, BIC, kernel_list, kernels_name, GPy=False
):
    """
    Update the BEST_MODELS dictionnary if the specific input model has a higher BIC score
    """
    if BIC > BEST_MODELS["score"] and BIC != float("inf"):
        BEST_MODELS["model_name"] = kernels_name
        BEST_MODELS["model_list"] = kernel_list
        BEST_MODELS["score"] = BIC
        if not GPy:
            BEST_MODELS["model"], BEST_MODELS["init_values"] = (
                model,
                model.initialisation_values,
            )
        else:
            wrapper = GPyWrapper(model, kernel_list)
            BEST_MODELS["model"] = wrapper
            BEST_MODELS["init_values"] = model.param_array
    return BEST_MODELS


def update_best_model_after_parallelized_step(outputs_threadpool, BEST_MODELS):
    for element in outputs_threadpool:
        if element is None:
            return BEST_MODELS
        else:
            if element["score"] > BEST_MODELS["score"]:
                BEST_MODELS = element
    del outputs_threadpool
    return BEST_MODELS


def log_cholesky_l_test(X, Y, params, kernel):
    """
        Compute negative log-likelihood using cholesky decomposition
    inputs :
        X tf Tensor, training x
        Y tf Tensor, training data y
        params list, vector containing model's params
        kernel list, list containg model's kernel and theirs operations
    outputs:
        loss float64, negative log likelihood
    """
    num = 0
    params_name = list(params.keys())
    cov = 1
    for op in kernel:
        if op[0] == "+":
            method = KERNELS_FUNCTIONS[op[1:]]
            par = params_name[num : num + KERNELS_LENGTH[op[1:]]]
            if not method:
                raise NotImplementedError("Method %s not implemented" % op[1:])
            cov += method(X, X, [params[p] for p in par])
            num += KERNELS_LENGTH[op[1:]]
        elif op[0] == "*":
            method = KERNELS_FUNCTIONS[op[1:]]
            par = params_name[num : num + KERNELS_LENGTH[op[1:]]]
            if not method:
                raise NotImplementedError("Method %s not implemented" % op[1:])
            cov = tf.math.multiply(cov, method(X, X, [params[p] for p in par]))
            num += KERNELS_LENGTH[op[1:]]
        elif op[0] == "C":
            kernel_list = op[3:-1].replace(" ", "").split(",")
            left_method = KERNELS_FUNCTIONS[kernel_list[0][2:-1]]
            par_name_left_method = params_name[
                num : num + KERNELS_LENGTH[kernel_list[0][2:-1]]
            ]
            num += KERNELS_LENGTH[kernel_list[0][2:-1]]
            right_method = KERNELS_FUNCTIONS[kernel_list[1][2:-1]]
            par_name_right_method = params_name[
                num : num + KERNELS_LENGTH[kernel_list[1][2:-1]]
            ]
            num += KERNELS_LENGTH[kernel_list[1][2:-1]]
            par_name_sigmoid, num = params_name[num : num + 2], num + 2
            cov += CP(
                X,
                X,
                [params[p] for p in par_name_sigmoid],
                left_method,
                right_method,
                [params[p] for p in par_name_left_method],
                [params[p] for p in par_name_right_method],
            )
    decomposed, _jitter, loop = False, 10e-7, 0
    try:
        _L = tf.cast(
            tf.linalg.cholesky(
                tf.cast(
                    cov
                    + (params["noise"] + _jitter)
                    * tf.eye(X.shape[0], dtype=_precision),
                    dtype=_precision,
                )
            ),
            dtype=_precision,
        )
    except Exception as e:
        print(e)
    _temp = tf.cast(tf.linalg.solve(_L, Y), dtype=_precision)
    alpha = tf.cast(tf.linalg.solve(tf.transpose(_L), _temp), dtype=_precision)
    loss = (
        0.5 * tf.cast(tf.matmul(tf.transpose(Y), alpha), dtype=_precision)
        + tf.cast(tf.math.log(tf.linalg.det(_L)), dtype=_precision)
        + 0.5 * tf.cast(X.shape[0] * tf.math.log([PI * 2]), dtype=_precision)
    )
    return loss


def train_step(
    model,
    iteration,
    X_train,
    Y_train,
    kernels_name,
    OPTIMIZER=tf.optimizers.Adamax(learning_rate=0.06),
):
    """
        Single step of training using first ordre stochastic gradient descent
    inputs
        model CustomModel object
        iteration int itération counter non used
        X_train tf Tensor,
        Y_train tf Tensor,
        kernels_names list , list of kernels of the model
        OPTIMIZER tf.optimizers
    outputs
        val float, objective function value
    """
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(model.variables)
        val = model(X_train, Y_train, kernels_name)
    gradient = tape.gradient(val, model.variables)
    try:
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e:
        OPTIMIZER.apply_gradients(gradient, model.variables)
    return val


def train_step_single(
    model,
    iteration,
    X_train,
    Y_train,
    kernels_name,
    OPTIMIZER=tf.optimizers.Adamax(learning_rate=0.06),
):
    """
        Single step of training using first ordre stochastic gradient descent
    inputs
        model CustomModel object
        iteration int itération counter non used
        X_train tf Tensor,
        Y_train tf Tensor,
        kernels_names list , list of kernels of the model
        OPTIMIZER tf.optimizers
    outputs
        val float, objective function value
    """
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(model.variables)
        val = model(X_train, Y_train)
    gradient = tape.gradient(val, model.variables)
    try:
        OPTIMIZER.apply_gradients(zip(gradient, model.variables))
    except Exception as e:
        OPTIMIZER.apply_gradients(gradient, model.variables)
    return val


def whitenning_datas(X):
    mean, var = tf.nn.moments(X, axes=[0])
    X = (X - mean) / var
    return X


""" function factory is a adapted version of  :
    https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
"""


def function_factory(model, loss_f, X, Y, params, kernel):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model._opti_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model._opti_variables[i].assign(
                tf.cast(tf.reshape(param, shape), dtype=_precision)
            )

    # now create a function that will be returned by this factory

    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = model(X, Y, kernel)[0][0]

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.variables)
        grads = tf.dynamic_stitch(idx, grads)

        # print out iteration & loss
        # f.iter.assign_add(1)
        # tf.print("Iter:", f.iter, "loss:", loss_value)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])
        # return loss_value ,grads
        return np.array(loss_value, order="F"), np.array(grads, order="F")

    # store these information as members so we can use them outside the scope
    f.iter = tf.convert_to_tensor(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f
