import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
from utils import *
import os 
import kernels
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float32')
PI = m.pi

KERNELS_FUNCTIONS = {
    "LIN" : kernels.LIN,
    "WN" : kernels.WN,
    "PER" : kernels.PER,
    "SE" : kernels.SE,

}

class PeriodicRegressor(object):
    def __init__(self):
        self._l = tf.compat.v1.get_variable('l',
                   dtype=tf.float32,
                   shape=(1,),
                   initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._p = tf.compat.v1.get_variable('p',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._sigma = tf.compat.v1.get_variable('sigma',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))


    @tf.function
    def __call__(self,X_train,Y_train):
        params={"l":self._l,"p":self._p,"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="PER")


    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = PER(X_train,X_train,l=self._l,p=self._p,sigma=self._sigma)
        cov_ss =  PER(X_s,X_s,l=self._l,p=self._p,sigma=self._sigma)
        cov_s  = PER(X_train,X_s,l=self._l,p=self._p,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._l,self._p,self._sigma
    
    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))


class LinearRegressor(object) :

    def __init__(self) :
        self._c = tf.compat.v1.get_variable('c',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=5.))
                   
    @tf.function
    def __call__(self,X_train,Y_train):
        params={"c":self._c}
        return log_cholesky_l(X_train,Y_train,params,kernel="LIN")

    @property
    def variables(self):
        return self._c


    def predict(self,X_train,Y_train,X_s):
        cov = kernels.LIN(X_train,X_train,c=self._c)
        cov_ss =  kernels.LIN(X_s,X_s,c=self._c)
        cov_s  = kernels.LIN(X_train,X_s,c=self._c)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))



class SquaredExpRegressor(object) :

    def __init__(self) :
        self._l = tf.compat.v1.get_variable('l',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=100.))
        self._sigma = tf.compat.v1.get_variable('sigmab',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=100.))
                    

    @tf.function
    def __call__(self,X_train,Y_train):
        params={"l":self._l,"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="SE")

    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = exp(X_train,X_train,l=self._l,sigma=self._sigma)
        cov_ss =  exp(X_s,X_s,l=self._l,sigma=self._sigma)
        cov_s  = exp(X_train,X_s,l=self._l,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._l,self._sigma

    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))
        

class WhiteNoiseRegressor(object) :

    def __init__(self) :
        self._sigma = tf.compat.v1.get_variable('sigma',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._N = tf.compat.v1.get_variable('unused_variable',
                    dtype=tf.float32,
                    initializer=0.0)
                    
    @tf.function
    def __call__(self,X_train,Y_train):
        params={"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="WN")


    def predict(self,X_train,Y_train,X_s):
        cov = WhiteNoise(X_train,X_train,sigma=self._sigma)
        cov_ss =  WhiteNoise(X_s,X_s,sigma=self._sigma)
        cov_s  = WhiteNoise(X_train,X_s,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._sigma,self._N


    def viewVar(self):
        list_vars = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        for var in list_vars : 
            print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()))))+"{}".format(var.numpy()))


class CustomModel(object):
    def __init__(self,params):
        for attr in params.keys() :
            pars = params[attr]
            for var in pars :
                self.__dict__[var] = tf.compat.v1.get_variable(var,
                        dtype=tf.float32,
                        shape=(1,),
                        initializer=tf.random_uniform_initializer(minval=1., maxval=1.))
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
    

    @tf.function
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
        cov = self._get_cov(X_train,X_train,kernels_name,params)
        cov_ss =  self._get_cov(X_s,X_s,kernels_name,params)
        cov_s  =  self._get_cov(X_train,X_s,kernels_name,params)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
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
        n =tf.Variable(X_train.shape[0],dtype=tf.float32)
        k = tf.Variable(len(kernels_name),dtype=tf.float32)
        ll = log_cholesky_l_test(X_train,Y_train,params,kernel=kernels_name)
        return k*tf.math.log(n) + 2*ll

    def plot(self,mu,cov,X_train,Y_train,X_s,kernel_name="None"):
        mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=100)
        if kernel_name is not None : plt.title("kernel :"+''.join(kernel_name)[1:])
        plot_gs_pretty(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
        plt.show()

        