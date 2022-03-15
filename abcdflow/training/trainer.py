import os
import logging
import contextlib
import functools
import pickle
import sys
import itertools
from itertools import chain
import time
import math as m

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import GPy
import pandas as pd
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import scipy
from gpflow.utilities import print_summary
from termcolor import colored

from ..kernels.kernels_utils import KERNELS, KERNELS_LENGTH, KERNELS_OPS, GPY_KERNELS
from .search import *
from ..regressors.Regressor import *
from ..regressors.Regressor_GPy import *
from ..kernels.kernels_utils import *
from .training_utils import *
from ..plots.plotting_utils import *

logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.backend.set_floatx("float64")

PI = m.pi
_precision = tf.float64
config = tf.compat.v1.ConfigProto(device_count={"GPU": 0, "CPU": 4})
config.gpu_options.allow_growth = True
config.inter_op_parallelism_threads = 3
config.intra_op_parallelism_threads = 3
session = tf.compat.v1.Session(config=config)
borne = -1 * 10e40


class Trainer:
    def __init__(
        self,
        X_train,
        Y_train,
        X_s,
        nb_restart=15,
        nb_iter=2,
        do_plot=False,
        save_model=False,
        prune=False,
        OPTIMIZER=tf.optimizers.Adam(0.001),
        parralelize_code=False,
        verbose=False,
        nb_by_step=None,
        loop_size=10,
        experimental_multiprocessing=False,
        reduce_data=False,
        straigth=True,
        depth=5,
        initialisation_restart=5,
        GPY=False,
        mode="lfbgs",
        use_changepoint=False,
        base_kernels=["+PER", "+LIN", "+SE"],
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_s = X_s
        self.nb_restart = nb_restart
        self.nb_iter = nb_iter
        self.do_plot = do_plot
        self.save_model = save_model
        self.prune = prune
        self.OPTIMIZER = OPTIMIZER
        self.verbose = verbose
        self.nb_by_step = nb_by_step
        self.loop_size = loop_size
        self.experimental_multiprocessing = experimental_multiprocessing
        self.reduce_data = reduce_data
        self.straigth = straigth
        self.depth = depth
        self.initialisation_restart = initialisation_restart
        self.GPY = GPY
        self.mode = mode
        self.use_changepoint = use_changepoint
        self.base_kernels = base_kernels
        self.parralelize_code = parralelize_code

    def launch_analysis(self):
        """
            Main function wich launch the search in the model space
        inputs :

        outputs:
            model : CustomModel object, best model
            kernels : list, array of best model
        """
        if self.prune and self.nb_by_step is None:
            raise ValueError("As prune is True you need to precise nb_by_step")
        if self.nb_by_step is not None and self.nb_by_step > self.loop_size:
            raise ValueError("Loop size must be superior to nb_by_step")
        if self.straigth:
            print("You chooosed straightforward training")
        self.X_train, self.Y_train, self.X_s = (
            tf.convert_to_tensor(self.X_train, dtype=_precision),
            tf.convert_to_tensor(self.Y_train, dtype=_precision),
            tf.convert_to_tensor(self.X_s, dtype=_precision),
        )
        if self.reduce_data:
            self.X_train = whitenning_datas(self.X_train)
            self.Y_train = whitenning_datas(self.Y_train)
            self.X_s = whitenning_datas(self.X_s)
        i = -1
        try:
            if self.experimental_multiprocessing:
                model, kernels = self.multithreaded_straight_analysis(i)
            elif self.straigth:
                model, kernels = self.monothreaded_straight_analysis(i)
            elif not self.experimental_multiprocessing:
                model, kernels = self.griddy_search(i)
            return model, kernels
        except Exception as e:
            print(
                colored(
                    "The algorithm couldn't converge, please try normallizing data with the reduce_data parameter",
                    "red",
                )
            )
            return None, None

    def griddy_search(self, i):
        model, kernels = self.analyse(i)
        name = "./best_models/best_model"
        if self.do_plot:
            try:
                mu, cov = model.predict(self.X_train, self.Y_train, self.X_s, kernels)
                model.plot(mu, cov, self.X_train, self.Y_train, self.X_s, kernels)
            except:
                model.plot()
            plt.show()
        if self.save_model:
            with open(name, "wb") as f:
                pickle.dump(model, f)
            with open("./best_models/kernels", "wb") as f:
                pickle.dump(kernels, f)
        return model, kernels

    def monothreaded_straight_analysis(self, i):
        model, kernels = self.straigth_analyse(i)
        name = "./best_models/best_model"
        if self.do_plot:
            try:
                mu, cov = model.predict(self.X_train, self.Y_train, self.X_s, kernels)
                model.plot(mu, cov, self.X_train, self.Y_train, self.X_s, kernels)
            except:
                model.plot()
            plt.show()
        if self.save_model:
            with open(name, "wb") as f:
                pickle.dump(model, f)
            with open("./best_models/kernels", "wb") as f:
                pickle.dump(kernels, f)
        return model, kernels

    def multithreaded_straight_analysis(self, i):
        print("This is experimental, the speed may varie a lot !")
        self.parralelize_code = True
        try:
            model, kernels = self.straigth_analyse(i)
            name = "./best_models/best_model"
            if self.do_plot:
                try:
                    mu, cov = model.predict(
                        self.X_train, self.Y_train, self.X_s, kernels
                    )
                    model.plot(mu, cov, self.X_train, self.Y_train, self.X_s, kernels)
                except:
                    model.plot()
                plt.show()
            if self.save_model:
                with open(name, "wb") as f:
                    pickle.dump(model, f)
                with open("./best_models/kernels", "wb") as f:
                    pickle.dump(kernels, f)
        except Exception as e:
            print("failed to converge")
        return model, kernels

    def analyse(self, i):
        """
            Compare models for each step of the training, and keep the best model
        inputs :
            i :  int;  kernel position in the list
        outputs:
            model : CustomModel object, best model
            BEST_MODELS["model_list"] : list, array of best model
        """
        if i == -1:
            COMB = search(
                "", [], True, self.depth, self.use_changepoint, self.base_kernels
            )
        else:
            name = "search/model_list_" + str(i)
            with open(name, "rb") as f:
                COMB = pickle.load(f)
        kernels_name, kernel_list = "", []
        BEST_MODELS = {
            "model_name": [],
            "model_list": [],
            "model": [],
            "score": borne,
            "init_values": None,
        }
        TEMP_BEST_MODELS = pd.DataFrame(columns=["Name", "score"])
        iteration = 0
        j = 0
        full_length = len(COMB)
        loop_size = 24
        while len(COMB) > 1:
            try:
                combi = COMB[j]
            except Exception as e:
                break
            iteration += 1
            j += 1
            if self.prune:
                if iteration % loop_size == 0:
                    j = 0
                    TEMP_BEST_MODELS = TEMP_BEST_MODELS[: self.nb_by_step]
                    _before_len = len(COMB)
                    COMB = prune(TEMP_BEST_MODELS["Name"].tolist(), COMB[iteration:])
                    _to_add = _before_len - len(COMB) - 1
                    iteration += _to_add
                BEST_MODELS, TEMP_BEST_MODELS = self.search_step(
                    combi, BEST_MODELS, TEMP_BEST_MODELS
                )
                TEMP_BEST_MODELS = TEMP_BEST_MODELS.sort_values(
                    by=["score"], ascending=True
                )[: self.nb_by_step]
                print_trainning_steps(iteration, full_length, combi)
            else:
                COMB, j = COMB[1:], 0
                BEST_MODELS = self.search_step(combi, BEST_MODELS, TEMP_BEST_MODELS)
                print_trainning_steps(iteration, full_length, combi)
            sys.stdout.write("\n")
            sys.stdout.flush()
        if not self.GPY:
            model = BEST_MODELS["model"]
            model.view_parameters(BEST_MODELS["model_list"])
            print(
                colored("[STATE]", "red"),
                "model BIC is {}".format(
                    model.compute_BIC(
                        self.X_train, self.Y_train, BEST_MODELS["model_list"]
                    )
                ),
            )
        return BEST_MODELS["model"], BEST_MODELS["model_list"]

    def search_step(
        self, combi, BEST_MODELS, TEMP_BEST_MODELS, unique=False, single=False
    ):
        """
            Launch the training of a single gaussian process : Create model, initialize params with the params of the previous best models, launch the optimization
            compute BIC and update the best models array if BIC > best_model's score
        inputs :
            BEST_MODELS : dict, dictionnary containing the best model and it score
            TEMP_BEST_MODELS :  dict, dictionnary containing temporaries bests models and theirs score
        outputs:
            BEST_MODELS : dict, dictionnary containing the best model and it score
        """
        j = 0
        lr = 0.1
        init_values = BEST_MODELS["init_values"]
        try:
            if not unique:
                kernel_list = list(combi)
            else:
                kernel_list = list([combi])
            if single:
                kernel_list = combi
            kernels_name = "".join(combi)
            true_restart = 0
            kernels = preparekernel(kernel_list)
            failed = 0
            if kernels_name[0] != "*":
                if not self.GPY:
                    while (
                        true_restart < self.initialisation_restart
                        and failed < self.initialisation_restart * 2
                    ):
                        try:
                            model = CustomModel(kernels, init_values, self.X_train)
                            model = self.train(model, kernel_list)
                            BIC = model.compute_BIC(
                                self.X_train, self.Y_train, kernel_list
                            )
                            if self.verbose:
                                print("[STATE] ", BIC)
                            BEST_MODELS = update_current_best_model(
                                BEST_MODELS, model, BIC, kernel_list, kernels_name
                            )
                            if BIC > BEST_MODELS["score"] and self.prune:
                                TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS) + 1] = [
                                    [kernels_name],
                                    float(BIC[0][0]),
                                ]
                            if m.isnan(BIC) == False or m.isinf(BIC) == False:
                                true_restart += 1
                            del model
                        except Exception as e:
                            failed += 1
                else:
                    k = gpy_kernels_from_names(kernel_list)
                    model = GPy.models.GPRegression(
                        self.X_train.numpy(), self.Y_train.numpy(), k, normalizer=False
                    )
                    model.optimize_restarts(self.initialisation_restart)
                    BIC = -model.objective_function()
                    BEST_MODELS = update_current_best_model(
                        BEST_MODELS, model, BIC, kernel_list, kernels_name, GPy=True
                    )
                    if BIC > BEST_MODELS["score"] and self.prune:
                        TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS) + 1] = [
                            [kernels_name],
                            BIC,
                        ]
        except Exception as e:
            print(e)
            print(colored("[ERROR]", "red"), " error with kernel :", kernels_name)
        if self.prune:
            return BEST_MODELS, TEMP_BEST_MODELS
        else:
            return BEST_MODELS

    def straigth_analyse(self, i):
        """
            Find best combinaison of kernel that descrive the training data , keep one best model at each step contrary to the greedy seach of the analyse function
        inputs :
            i :  int;  kernel position in the list
        outputs:
            model : CustomModel object, best model
            kernel : list, array of best model
        """
        BEST_MODELS = {
            "model_name": [],
            "model_list": [],
            "model": [],
            "score": borne,
            "init_values": None,
        }
        count, constant, old_model = 0, 0, ""
        train_length = (self.depth + 1) * len(KERNELS) + len(KERNELS)
        COMB = first_kernel(self.use_changepoint, self.base_kernels)
        print(colored("[STATE]", "red"), " starting with ", COMB)
        for loop in range(self.depth):
            TEMP_BEST_MODELS = pd.DataFrame(columns=["Name", "score"])
            loop += 1
            if loop > 1:
                # COMB = search_and_add(tuple(BEST_MODELS["model_list"]),use_changepoint)
                COMB = search_and_add(
                    tuple(BEST_MODELS["model_list"]), False, self.base_kernels
                )
                print(
                    colored("[STATE]", "red"),
                    " Next combinaison to try : {}".format(COMB),
                )
            iteration, j = 0, 0
            if self.parralelize_code:
                outputs = self.parralelize(COMB, BEST_MODELS)
                BEST_MODELS = update_best_model_after_parallelized_step(
                    outputs, BEST_MODELS
                )
                print(
                    colored("[INTERMEDIATE RESULTS] ", "green"),
                    "The best model is {} at layer {}".format(
                        BEST_MODELS["model_list"], loop
                    ),
                )
                if old_model == BEST_MODELS["model_name"]:
                    constant += 1
                old_model = BEST_MODELS["model_name"]
                if constant > 1:
                    return BEST_MODELS["model"], BEST_MODELS["model_list"]
                if loop > 2:
                    new_COMB = replacekernel(BEST_MODELS["model_list"])  # swap step
                    print(
                        colored("[STATE]", "red"),
                        " Trying to switch kernels : trying {} ".format(new_COMB),
                    )
                    outputs = self.parralelize(COMB, BEST_MODELS)
                    BEST_MODELS = update_best_model_after_parallelized_step(
                        outputs, BEST_MODELS
                    )
            else:
                while j < len(COMB):
                    try:
                        combi = COMB[j]
                    except Exception as e:
                        break
                    iteration += 1
                    j += 1
                    count += 1
                    TEMP_MODELS = self.search_step(combi, BEST_MODELS, TEMP_BEST_MODELS)
                    print_trainning_steps(count, train_length, combi)
                if TEMP_MODELS["score"] > BEST_MODELS["score"]:
                    BEST_MODELS = TEMP_MODELS
                print(
                    colored("[INTERMEDIATE RESULTS] ", "green"),
                    "The best model is {} at layer {}".format(
                        BEST_MODELS["model_list"], loop
                    ),
                )
                if loop > 2 and len(BEST_MODELS["model_list"]) > 2:
                    new_COMB = replacekernel(BEST_MODELS["model_list"])  # swap step
                    print(
                        colored("[STATE]", "red"),
                        " Trying to switch kernels : trying {} ".format(new_COMB),
                    )
                    j = 0
                    while j < len(new_COMB):
                        try:
                            combi = new_COMB[j]
                        except Exception as e:
                            break
                        j += 1
                        TEMP_MODELS = self.search_step(
                            combi, BEST_MODELS, TEMP_BEST_MODELS
                        )
                        print_trainning_steps(count, train_length, combi)
                    if TEMP_MODELS["score"] > BEST_MODELS["score"]:
                        BEST_MODELS = TEMP_MODELS
            print(
                colored("[STATE]", "red"),
                " The best model is {} after swap ".format(BEST_MODELS["model_list"]),
            )
        if not self.GPY and self.verbose:
            model = BEST_MODELS["model"]
            model.view_parameters(BEST_MODELS["model_list"])
            print(
                colored("[STATE]", "red"),
                " model BIC is {}".format(
                    model.compute_BIC(
                        self.X_train, self.Y_train, BEST_MODELS["model_list"]
                    )
                ),
            )
        else:
            model = BEST_MODELS["model"]
            BEST_MODELS["model"].view_parameters(BEST_MODELS["model_list"])
        return BEST_MODELS["model"], BEST_MODELS["model_list"]

    def parralelize(self, combi, BEST_MODELS):
        """
        Use class multiprocessing.dummies to multithread one step train , keep all the score of every models in order to compared them lately
        """
        params = self.generator_params(combi, BEST_MODELS)
        pool = ThreadPool(3)
        for parameter in params:
            outputs = pool.starmap(self.search_step_parrallele, params)
        pool.close()
        pool.join()
        del pool
        return outputs

    def generator_params(self, combi, BEST_MODELS):
        for comb in combi:
            yield [
                self.X_train,
                self.Y_train,
                self.X_s,
                comb,
                BEST_MODELS,
                self.nb_restart,
                self.nb_iter,
                self.nb_by_step,
                self.prune,
                self.verbose,
                self.OPTIMIZER,
                self.mode,
                False,
                False,
                self.initialisation_restart,
                self.GPY,
            ]

    def search_step_parrallele(
        self,
        X_train,
        Y_train,
        X_s,
        combi,
        TEMP_BEST_MODELS,
        nb_restart,
        nb_iter,
        nb_by_step,
        prune,
        verbose,
        OPTIMIZER,
        mode="lfbgs",
        unique=False,
        single=False,
        initialisation_restart=10,
        GPY=False,
    ):
        """
            Launch the training of a single gaussian process : Create model, initialize params with the params of the previous best models, launch the optimization
            compute BIC and keep the score in a dictionnary (not to use shared BEST_MODELS dictionnary variable)
        inputs :
            X_train : Tensor, Training X
            Y_train : Tensor, Training Y
            BEST_MODELS : dict, dictionnary containing the best model and it score
            TEMP_BEST_MODELS :  dict, dictionnary containing temporaries bests models and theirs score
            nb_iter : int, number of iterations during the training
            nb_by_step : int, number of best model to keep when prune is true
            nb_restart : int, retrain on same data (epoch)
            kernels_name : string, updated kernel name   ex +LIN*PER*PER
            OPTIMIZER : tf optimizer object
            verbose : Bool, print training process
            mode : string , training mode
            unique : Bool, if only one kernel in the list ex +PER
            single : Bool,
            initialisation_restart : int, number of restart training with different initiatlisation parameters
        outputs:
            BEST_MODELS : dict, dictionnary containing the best model and it score
        """
        true_restart = 0
        BEST_MODELS = {
            "model_name": [],
            "model_list": [],
            "model": [],
            "score": borne,
            "init_values": None,
        }
        init_values = TEMP_BEST_MODELS["init_values"]
        del TEMP_BEST_MODELS
        try:
            if not unique:
                kernel_list = list(combi)
            else:
                kernel_list = list([combi])
            if single:
                kernel_list = combi
            kernels_name = "".join(combi)
            true_restart = 0
            kernels = preparekernel(kernel_list)
            failed = 0
            if kernels_name[0] != "*":
                if not GPY:
                    while true_restart < initialisation_restart and failed < 20:
                        try:
                            model = CustomModel(kernels, init_values, X_train)
                            model = self.train(model, kernel_list)
                            BIC = model.compute_BIC(X_train, Y_train, kernel_list)
                            BEST_MODELS = update_current_best_model(
                                BEST_MODELS, model, BIC, kernel_list, kernels_name
                            )
                            true_restart += 1
                            del model
                        except Exception:
                            failed += 1
                else:
                    try:
                        k = gpy_kernels_from_names(kernel_list)
                        model = GPy.models.GPRegression(
                            X_train.numpy(), Y_train.numpy(), k, normalizer=False
                        )
                        model.optimize_restarts(initialisation_restart)
                        BIC = -model.objective_function()
                        BEST_MODELS = update_current_best_model(
                            BEST_MODELS, model, BIC, kernel_list, kernels_name, GPy=True
                        )
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)
            print(colored("[ERROR]", "red"), " error with kernel :", kernels_name)
        else:
            return BEST_MODELS

    def train(self, model, kernels_name):
        """
            Train the model according to the parameters
        inputs :
            model : CustomModel object , Gaussian process model
            kernel_list : string, updated kernel name   ex +LIN*PER*PER
        outputs:
            best_model : dict, dictionnary containing the best model and it score
        """
        best = -1 * borne
        loop, base_model = 0, model
        lr = 0.1
        old_val, val, lim = 0, 0, 1.5
        if self.mode == "SGD":
            while loop < self.nb_restart:
                try:
                    model = base_model
                    if self.verbose:
                        for iteration in range(0, self.nb_iter):
                            if loop > 10:
                                OPTIMIZER.learning_rate.assign(0.001)
                            val = train_step(
                                model,
                                iteration,
                                self.X_train,
                                self.Y_train,
                                kernels_name,
                                OPTIMIZER,
                            )
                            if np.isnan(val):
                                loop += 1
                                break
                            sys.stdout.write(
                                "\r"
                                + "=" * int(iteration / self.nb_iter * 50)
                                + ">"
                                + "."
                                * int((self.nb_iter - iteration) / self.nb_iter * 50)
                                + "|"
                                + " * log likelihood  is : {:.4f} at iteration : {:.0f} at epoch : {:.0f} / {:.0f} with lr of: {}".format(
                                    val[0][0], iteration, loop + 1, self.nb_restart, lr
                                )
                            )
                            sys.stdout.flush()
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        loop += 1
                    else:
                        for iteration in range(1, self.nb_iter):
                            val = train_step(
                                model,
                                iteration,
                                self.X_train,
                                self.Y_train,
                                kernels_name,
                            )
                        loop += 1
                except Exception as e:
                    loop += 1
                if val < best:
                    best = val
                    best_model = model
        elif "lfbgs":
            try:
                self.nb_iter = max(self.nb_iter, 100)
                func = function_factory(
                    model,
                    log_cholesky_l_test,
                    self.X_train,
                    self.Y_train,
                    model._opti_variables,
                    kernels_name,
                )
                init_params = tf.dynamic_stitch(func.idx, model._opti_variables)
                # train the model with L-BFGS-B solver
                bnds = define_boundaries(model, self.X_train)
                results = scipy.optimize.minimize(
                    fun=func,
                    x0=init_params,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=tuple(bnds),
                    options={"maxiter": self.nb_iter},
                )
                best_model = model
            except Exception as e:
                print(e)
        else:
            raise NotImplementedError(
                "Mode not available please choose between lfbgs or SBD"
            )
        return best_model
