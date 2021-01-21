import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')

PI = m.pi
_precision = tf.float64

def LIN(x,y,params):
    c = params[0]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(tf.math.subtract(x,c*tf.ones_like(x)))
    y1 = tf.math.subtract(y,c*tf.ones_like(y))
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    w = tf.math.multiply(y2,x2) 
    return w


def CONST(x,y,sigma):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    return sigma*tf.ones_like(x2)




def PER(x,y1,params):
    l,p,sigma = params[0],params[1],params[2]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = PI/p
    const_2 = 2*(-1/tf.math.square(l))
    w = sigma * tf.math.exp(const_2*tf.math.square(tf.math.sin(const_1*tf.math.abs(tf.math.subtract(x2,y2)))))
    return w

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


def RQ(x,y,params):
    l,sigma,alpha = params[0],params[1],params[2]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y, multiply_y))
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const = tf.cast(2*alpha*l,dtype=_precision)
    power_matrix = -1*alpha*tf.ones_like(x2,dtype=_precision)
    const_matrix = const * tf.ones_like(x2,dtype=_precision)
    a = tf.cast(1/const*tf.math.add(tf.math.square(tf.math.subtract(y2,x2)),const_matrix),dtype=_precision)
    w = tf.math.pow(a, power_matrix, name="Powered_matrix")
    return sigma*w


