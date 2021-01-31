import numpy as np 
import tensorflow as tf 
from pprint import pprint
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
import os 
import tensorflow_probability as tfp 



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')

PI = m.pi
_precision = tf.float64

@tf.function
def LIN(x,y,params):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    c,sigmav = params[0],params[1]
    x1 = tf.transpose(tf.math.subtract(x,c*tf.ones_like(x)))
    y1 = tf.math.subtract(y,c*tf.ones_like(y))
    y2 = tf.transpose(tf.tile(y1, tf.constant([1,x.shape[0]])))
    x2 = tf.transpose(tf.tile(x1, tf.constant([y.shape[0],1])))
    return sigmav*tf.math.multiply(y2,x2) 

@tf.function
def CONST(x,y,sigma):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    return sigma*tf.ones_like(x2)

@tf.function
def WN(x,y,sigma):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    x2 = tf.transpose(tf.tile(x1, tf.constant([y.shape[0],1])))
    y2 = tf.transpose(tf.tile(y, tf.constant([1,x.shape[0]])))
    w = tf.cast(tf.math.logical_not(tf.not_equal( tf.math.subtract(x2,y2), tf.constant(0, dtype=_precision)), name="inverse"),dtype=_precision)
    return sigma*w

@tf.function
def PER(x,y1,params):
    l,p,sigma = params[0],params[1],params[2]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same dimension"
    w = sigma * tf.math.exp(2*(-1/tf.math.square(l))*tf.math.square(tf.math.sin(PI/p*tf.math.abs(tf.math.subtract(tf.transpose(tf.tile(tf.transpose(x), tf.constant([y1.shape[0],1]))),tf.transpose(tf.tile(y1, tf.constant([1,x.shape[0]]))))))))
    return w

@tf.function
def SE(x,y1,params):
    l,sigma = params[0],params[1]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = 0.5*tf.cast(-1/tf.math.square(l),dtype=_precision)
    return sigma*tf.math.exp(tf.math.square(tf.math.subtract(y2,x2))*const_1)



@tf.function
def RQ(x,y,params):
    l,sigma,alpha = params[0],params[1],params[2]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    multiply_y = tf.constant([1,x.shape[0]])
    multiply_x = tf.constant([y.shape[0],1])
    a = tf.cast(1/const*tf.math.add(tf.math.square(tf.math.subtract(tf.transpose(tf.tile(y, multiply_y)), tf.transpose(tf.tile(tf.transpose(x), multiply_x)))),tf.cast(2*alpha*l,dtype=_precision) * tf.ones_like( tf.transpose(tf.tile(tf.transpose(x), multiply_x)),dtype=_precision)),dtype=_precision)
    w = tf.math.pow(a, -1*alpha*tf.ones_like(x2,dtype=_precision), name="Powered_matrix")
    return sigma*w


# Still to do, implement sigmoid kernel
@tf.function
def CP(x,y,params,inverse=False):
    l,s = params[0],params[1]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_x = tf.constant([y.shape[0],1])
    multiply_y = tf.constant([1,x.shape[0]])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    y2 = tf.transpose(tf.tile(y, multiply_y))
    const = tf.cast(l/s,dtype=_precision)
    temp = const*tf.ones_like(x2)
    if not inverse : 
        sigmax = 0.5*(1+tf.math.tanh(tf.math.substract(const-x2))) 
        sigmay = 0.5*(1+tf.math.tanh(tf.math.subtract(const-y2))) 
        return tf.math.matmul(sigmax,sigmay)
    else :
        sigmax = tf.math.substract(temp,0.5*(1+tf.math.tanh(tf.math.substract(const-x2))))
        sigmay = tf.math.substract(temp,0.5*(1+tf.math.tanh(tf.math.subtract(const-y2))))
        return tf.math.matmul(sigmax,sigmay)