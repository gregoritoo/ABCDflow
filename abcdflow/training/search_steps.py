import math as m


import GPy
from termcolor import colored

from ..kernels.kernels_utils import KERNELS, KERNELS_LENGTH, KERNELS_OPS, GPY_KERNELS
from .search import *
from ..regressors.Regressor import *
from ..regressors.Regressor_GPy import *
from ..kernels.kernels_utils import *
from .training_utils import *
from ..plots.plotting_utils import *
from .training_utils import train, update_current_best_model


def search_step(
    X_train,
    Y_train,
    X_s,
    combi,
    BEST_MODELS,
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
    initialisation_restart=5,
    GPY=False,
):
    """
        Launch the training of a single gaussian process : Create model, initialize params with the params of the previous best models, launch the optimization
        compute BIC and update the best models array if BIC > best_model's score
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
        failed_counter = 0
        if kernels_name[0] != "*":
            if not GPY:
                while (
                    true_restart < initialisation_restart
                    and failed_counter < initialisation_restart * 2
                ):
                    try:
                        kernel_training_step(
                            X_train,
                            Y_train,
                            BEST_MODELS,
                            TEMP_BEST_MODELS,
                            nb_restart,
                            nb_iter,
                            prune,
                            verbose,
                            OPTIMIZER,
                            kernels,
                            init_values,
                            kernel_list,
                            kernels_name,
                            mode="lfbgs",
                        )
                    except Exception as e:
                        failed_counter += 1
            else:
                k = gpy_kernels_from_names(kernel_list)
                model = GPy.models.GPRegression(
                    X_train.numpy(), Y_train.numpy(), k, normalizer=False
                )
                model.optimize_restarts(initialisation_restart)
                BIC = -model.objective_function()
                BEST_MODELS = update_current_best_model(
                    BEST_MODELS, model, BIC, kernel_list, kernels_name, GPy=True
                )
                if BIC > BEST_MODELS["score"] and prune:
                    TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS) + 1] = [
                        [kernels_name],
                        BIC,
                    ]
    except Exception as e:
        print(colored("[ERROR]", "red"), " error with kernel :", kernels_name)
    if prune:
        return BEST_MODELS, TEMP_BEST_MODELS
    else:
        return BEST_MODELS


def kernel_training_step(
    X_train,
    Y_train,
    BEST_MODELS,
    TEMP_BEST_MODELS,
    nb_restart,
    nb_iter,
    prune,
    verbose,
    OPTIMIZER,
    kernels,
    init_values,
    kernel_list,
    kernels_name,
    mode="lfbgs",
):
    model = CustomModel(kernels, init_values, X_train)
    model = train(
        model,
        nb_iter,
        nb_restart,
        X_train,
        Y_train,
        kernel_list,
        OPTIMIZER,
        verbose,
        mode=mode,
    )
    BIC = model.compute_BIC(X_train, Y_train, kernel_list)
    if verbose:
        print("[STATE] ", BIC)
    BEST_MODELS = update_current_best_model(
        BEST_MODELS, model, BIC, kernel_list, kernels_name
    )
    if BIC > BEST_MODELS["score"] and prune:
        TEMP_BEST_MODELS.loc[len(TEMP_BEST_MODELS) + 1] = [
            [kernels_name],
            float(BIC[0][0]),
        ]
    if m.isnan(BIC) == False or m.isinf(BIC) == False:
        true_restart += 1
    del model
    return BEST_MODELS
