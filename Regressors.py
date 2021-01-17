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
tf.keras.backend.set_floatx('float32')
PI = m.pi


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
        self._noise = tf.compat.v1.get_variable('noise',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))

    @tf.function
    def __call__(self,X_train,Y_train):
        params={"l":self._l,"p":self._p,"sigma":self._sigma,"noise":self._noise}
        return log_cholesky_l(X_train,Y_train,params,kernel="Periodic")


    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = Periodic(X_train,X_train,l=self._l,p=self._p,sigma=self._sigma)
        cov_ss =  Periodic(X_s,X_s,l=self._l,p=self._p,sigma=self._sigma)
        cov_s  = Periodic(X_train,X_s,l=self._l,p=self._p,sigma=self._sigma)
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
        self._c = tf.compat.v1.get_variable('l',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
                   
        self._sigmav = tf.compat.v1.get_variable('sigmav',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=0.01, maxval=10.))
    @tf.function
    def __call__(self,X_train,Y_train):
        
        params={"c":self._c,"sigmav":self._sigmav}
        return log_cholesky_l(X_train,Y_train,params,kernel="Linear")

    @property
    def variables(self):
        return self._c,self._sigmav

    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = Linear(X_train,X_train,c=self._c,sigmav=self._sigmav)
        cov_ss =  Linear(X_s,X_s,c=self._c,sigmav=self._sigmav)
        cov_s  = Linear(X_train,X_s,c=self._c,sigmav=self._sigmav)
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
                    
    @tf.function
    def __call__(self,X_train,Y_train):
        params={"sigma":self._sigma}
        return log_cholesky_l(X_train,Y_train,params,kernel="WN")

    @tf.function
    def predict(self,X_train,Y_train,X_s):
        cov = WhiteNoise(X_train,X_train,sigma=self._sigma)
        cov_ss =  WhiteNoise(X_s,X_s,sigma=self._sigma)
        cov_s  = WhiteNoise(X_train,X_s,sigma=self._sigma)
        mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
        return mu,cov

    @property
    def variables(self):
        return self._sigma


    def viewVar(self):
        var = self.variables
        print("Parameters : ")
        print("   var name               |               value")
        print("   {}".format(str(var.name))+" "*int(23-int(len(str(var.name))))+"|"+" "*int(23-int(len(str(var.numpy()[0]))))+"{}".format(var.numpy()[0]))