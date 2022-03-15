import os
import logging
from typing import *
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import seaborn as sn

from ..kernels.kernels_utils import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.backend.set_floatx("float64")
logging.getLogger("tensorflow").setLevel(logging.FATAL)
tf.get_logger().setLevel("INFO")

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"

PI = m.pi
OPTIMIZER = tf.optimizers.Adamax(learning_rate=0.06)
_jitter = 1e-7
_precision = tf.float64


def make_df(X: List, stdp: List, stdi: List) -> pd.DataFrame:
    X = np.array(X).reshape(-1)
    Y = np.array(np.arange(len(X))).reshape(-1)
    stdp = np.array(stdp).reshape(-1)
    stdi = np.array(stdi).reshape(-1)
    df = pd.DataFrame({"x": X, "y": Y, "stdp": stdp, "stdi": stdi}, index=Y)
    return df


def plot_gs_pretty(
    true_data: tf.Tensor,
    mean: tf.Tensor,
    X_train: tf.Tensor,
    X_s: tf.Tensor,
    stdp: tf.Tensor,
    stdi: tf.Tensor,
    color: str = "blue",
) -> int:
    plt.style.use("seaborn-dark-palette")
    try:
        true_data, X_train, X_s = true_data.numpy(), X_train.numpy(), X_s.numpy()
        mean, stdp, stdi = mean.numpy(), stdp.numpy(), stdi.numpy()
    except Exception as e:
        pass
    true_data, X_train, X_s = (
        true_data.reshape(-1),
        X_train.reshape(-1),
        X_s.reshape(-1),
    )
    mean, stdp, stdi = mean.reshape(
        -1), stdp.reshape(-1), stdi.reshape(-1).reshape(-1)
    plt.style.use("seaborn-dark-palette")
    plt.plot(X_s, mean, color="blue", label="Predicted values")
    plt.fill_between(X_s, stdp, stdi, facecolor=CB91_Blue,
                     alpha=0.2, label="Conf I")
    plt.plot(X_train, true_data, color="black", label="True data", marker="x")
    plt.legend()
    plt.show()
    return 0


def plot_gs(
    true_data: np.ndarray,
    mean: np.ndarray,
    X_train: np.ndarray,
    X_s: np.ndarray,
    stdp: np.ndarray,
    stdi: np.ndarray,
    color: str = "blue",
) -> int:
    plt.figure(figsize=(32, 16), dpi=100)
    print(X_s)
    plt.style.use("seaborn")
    plt.plot(X_s, mean, color="green", label="Predicted values")
    plt.fill_between(
        X_s.reshape(
            -1,
        ),
        stdp,
        stdi,
        facecolor=CB91_Blue,
        alpha=0.2,
        label="Conf I",
    )
    plt.plot(X_train, true_data, color=CB91_Amber, label="True data")
    plt.legend()
    return 0


def get_values(
    mu_s: np.ndarray, cov_s: np.ndarray, nb_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Get prediction using predicted mean and covariance function
    inputs
        mu_s array, predicted mean
        cov_s array, predicted covariance
        nb_samples int, number of sample to draw to estimate prediction
    outputs
        mean numpy array, predicted mean
        stdp numpy array, upper bound of 99% CI
        stdi numpy array, lower bound of 99% CI
    """
    samples = np.random.multivariate_normal(mu_s, cov_s, nb_samples)
    stdp = [
        np.mean(samples[:, i]) + 1.96 * np.std(samples[:, i])
        for i in range(samples.shape[1])
    ]
    stdi = [
        np.mean(samples[:, i]) - 1.96 * np.std(samples[:, i])
        for i in range(samples.shape[1])
    ]
    mean = [np.mean(samples[:, i]) for i in range(samples.shape[1])]
    return mean, stdp, stdi


def print_trainning_steps(
    count: int, train_length: int, combinaison_element: Tuple
) -> int:
    """
        Print the state of the training ex, ==>..|
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
    return 0
