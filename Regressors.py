import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from utils import *
import os 
from language import *
import kernels as kernels 
import itertools
from language import *
from search import preparekernel
from utils import KERNELS_FUNCTIONS,GPY_KERNELS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')
PI = m.pi
_precision = tf.float64



class CustomModel(object):
    def __init__(self,params,existing=None):
        for attr in params.keys() :
            pars = params[attr]
            for var in pars :
                if existing is  None :
                    self.__dict__[var] = tf.compat.v1.get_variable(var,
                            dtype=_precision,
                            shape=(1,),
                            initializer=tf.random_uniform_initializer(minval=1e-2, maxval=100.))
                else :
                    if var in existing.keys() :
                        self.__dict__[var] = tf.Variable(existing[var],dtype=_precision)
                    else :
                        self.__dict__[var] = tf.compat.v1.get_variable(var,
                            dtype=_precision,
                            shape=(1,),
                            initializer=tf.random_uniform_initializer(minval=1e-2, maxval=100.))
        self.__dict__["noise"] = tf.compat.v1.get_variable("noise",
                            dtype=_precision,
                            shape=(1,),
                            initializer=tf.random_uniform_initializer(minval=1e-2, maxval=100.))

    @property
    def initialisation_values(self):
        return dict({k:v for k,v in zip(list(vars(self).keys()),list(vars(self).values()))})

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
    

    #@tf.function
    def __call__(self,X_train,Y_train,kernels_name):
        params=vars(self)
        return log_cholesky_l_test(X_train,Y_train,params,kernel=kernels_name)

    def viewVar(self,kernels):
        list_vars = self.variables
        print("\n Parameters of  : {}".format(kernels))
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()))))+"{}".format(var.numpy()))


    def predict(self,X_train,Y_train,X_s,kernels_name):
        params= self._variables
        try :
            X_train = tf.Variable(X_train,dtype=_precision)
            Y_train = tf.Variable(Y_train,dtype=_precision)
            X_s = tf.Variable(X_s,dtype=_precision)
        except Exception as e:
            pass
        cov = self._get_cov(X_train,X_train,kernels_name,params)
        cov_ss =  self._get_cov(X_s,X_s,kernels_name,params)
        cov_s  =  self._get_cov(X_train,X_s,kernels_name,params)
        mu,cov = self._compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    def _get_cov(self,X,Y,kernel,params):
        params_name = list(params.keys())
        cov = 0
        num = 0
        for op in kernel :
            if op[0] == "+":
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov += method(X,Y,[params[p] for p in par])
                num += KERNELS_LENGTH[op[1:]]
            elif op[0] == "*":
                method = KERNELS_FUNCTIONS[op[1:]]
                par =params_name[num:num+KERNELS_LENGTH[op[1:]]]
                if not method:
                    raise NotImplementedError("Method %s not implemented" % op[1:])
                cov  = tf.math.multiply(cov,method(X,Y,[params[p] for p in par]))
                num += KERNELS_LENGTH[op[1:]]
        return cov


    def compute_BIC(self,X_train,Y_train,kernels_name):
        params= self._variables
        n =tf.Variable(X_train.shape[0],dtype=_precision)
        k = tf.Variable(len(kernels_name),dtype=_precision)
        ll = log_cholesky_l_test(X_train,Y_train,params,kernel=kernels_name)
        return  -ll - 0.5*k*tf.math.log(n)

    def plot(self,mu,cov,X_train,Y_train,X_s,kernel_name=None):
        try :
            Y_train,X_train,X_s = Y_train,X_train,X_s
        except Exception as e :
            print(e)
        mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)
        if kernel_name is not None : plt.title("kernel :"+''.join(kernel_name)[1:])
        plot_gs_pretty(Y_train,np.array(mean),X_train,X_s,np.array(stdp),np.array(stdi))
        plt.show()

    def _compute_posterior(self,y,cov,cov_s,cov_ss):
        params= self._variables
        mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+params["noise"]*tf.eye(cov.shape[0],dtype=_precision))),y)
        cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+params["noise"]*tf.eye(cov.shape[0],dtype=_precision))),cov_s)
        return mu,cov


    def split_params(self,kernel_list):
        params = list(self._opti_variables_name)
        list_params = []
        pos = 0
        for element in kernel_list :
            if element[1] == "P" : 
                list_params.append(params[pos:pos+3])
                pos+=3
            else : 
                list_params.append(params[pos:pos+2])
                pos+=2
        return list_params

    def describe(self,kernel_list) :
        list_params = self.split_params(kernel_list)
        params_dic = self._variables
        loop_counter= 0
        splitted,pos = devellopement(kernel_list)
        summary = "The signal has {} componants :\n".format(len(splitted))
        for element in splitted :
            summary =  comment(summary,element,pos[loop_counter],params_dic,list_params)  + "\n"
            loop_counter += 1
        summary = summary + "\t It also has a noise of {:.1f} .".format(self._variables["noise"].numpy()[0])
        print(summary)

    def decompose(self,kernel_list,X_train,Y_train,X_s):
        list_params = self.split_params(kernel_list)
        splitted,pos = devellopement(kernel_list)
        params_dic = self._variables
        loop_counter= 0
        cov = 0
        for element in splitted :
            kernels = preparekernel(element,scipy=True)
            list_of_dic = [list_params[position] for position in pos[loop_counter]]
            merged = list(itertools.chain(*list_of_dic))
            dictionary = dict(zip(merged, [params_dic[one] for one in merged]))
            decomp_model = CustomModel(kernels,dictionary)
            mu,cov = decomp_model.predict(X_train,Y_train,X_s,element)
            plt.title("kernel :"+''.join(element)[1:])
            decomp_model.plot(mu,cov,X_train,Y_train,X_s,kernel_name=None)
            plt.show(block=True)
            plt.close()

    
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
            kernels = preparekernel(element,scipy=True)
            list_of_dic = [list_params[position] for position in pos[loop_counter]]
            params = [list_params[position] for position in pos[loop_counter]]
            print(element)
            print(params)
            loop_counter += 1
            try :
                k = self._gpy_kernels_from_names(element,params)
            except Exception as e :
                print(e)
            print(kernel)
            model = GPy.models.GPRegression(X_train, Y_train, k, normalizer=False)
            model.plot()
            loop_counter += 1

    def _gpy_kernels_from_names(self,_kernel_list,params):
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
        return kernel