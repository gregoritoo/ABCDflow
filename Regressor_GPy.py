import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from kernels_utils import *
from training_utils import *
from plotting_utils import *
import os 
from language import *
import kernels as kernels 
import itertools
from language import *
from search import preparekernel,decomposekernel
from kernels_utils import KERNELS_FUNCTIONS
from termcolor import colored
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')
PI = m.pi
_precision = tf.float64


   
class GPyWrapper(object):
    def __init__(self,model,kernels):
        self._model = model 
        self._kernels_list = kernels

    def variables(self):
        return self._model.param_array

    def variables_names(self):
        return self._model.parameter_names()

    def viewVar(self,kernels) :
        print("\n Parameters of  : {}".format(kernels))
        print(self._model)

    def plot(self,mu=None,cov=None,X_train=None,Y_train=None,X_s=None,kernel_name=None):
        self._model.plot()
        plt.show()
        return 0

    def split_params(self,kernel_list,params_values):
        list_params = []
        pos = 0
        for element in kernel_list :
            if element[1] == "P" : 
                list_params.append(params_values[pos:pos+3])
                pos+=3
            elif element[1] == "L" : 
                list_params.append(params_values[pos:pos+1])
                pos+=1
            else :
                list_params.append(params_values[pos:pos+2])
                pos+=2
        return list_params
    
    def describe(self,kernel_list):
        splitted,pos = devellopement(kernel_list)
        loop_counter= 0
        variables_names = self.variables_names()
        variables = self.variables()
        list_params = self.split_params(kernel_list,variables)
        summary = "The signal has {} componants :\n".format(len(splitted))
        for element in splitted :
            summary =  comment_gpy(summary,element,pos[loop_counter],variables_names,list_params)  + "\n"
            loop_counter += 1
        summary = summary + "\t It also has a noise of {:.1f} .".format(variables[-1])
        print(summary)

    def viewVar(self,kernels):
        print(self._model)

    def decompose(self,kernel_list,X_train,Y_train,X_s):
        splitted,pos = devellopement(kernel_list)
        variables = self.variables()
        list_params = self.split_params(kernel_list,variables)
        loop_counter= 0
        counter = 0
        for element in splitted :
            kernels = preparekernel(element)
            list_of_dic = [list_params[position] for position in pos[loop_counter]]
            params = [list_params[position] for position in pos[loop_counter]]
            loop_counter += 1
            try :
                k = self._gpy_kernels_from_names(element,params)
            except Exception as e :
                print(e)
            model = GPy.models.GPRegression(X_train, Y_train, k, normalizer=False)
            model.plot()
            loop_counter += 1

"""    def _gpy_kernels_from_names(self,_kernel_list,params):
        try :
            kernel = GPy.kern.Linear(1,params[0][0])
        except Exception as e:
            print(e)
        print(kernel)
        for j in range(1,len(_kernel_list)) :
            if _kernel_list[j][0] == "+" :
                kernel = kernel + GPY_KERNELS[_kernel_list[j][1 :]](1,params[j])
            elif _kernel_list[j][0] == "*" :
                kernel = kernel * GPY_KERNELS[_kernel_list[j][1 :]](1,params[j])
            else :
                raise ValueError("Illicite operation")
        return kernel"""