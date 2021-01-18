import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
import sys 
tf.keras.backend.set_floatx('float32')

PI = m.pi



def LIN(x,y,params):
    c = params[0]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(tf.math.subtract(x,c*tf.ones_like(x)))
    y1 = tf.math.subtract(y,c*tf.ones_like(y))
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    w = tf.math.multiply(y2,x2) 
    return w


def WN(x,y,sigma):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(x)
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    return sigma*tf.ones_like(x2)




def PER(x,y1,params):
    l,p,sigma = params[0],params[1],params[2]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = PI/p
    const_2 = 0.5*tf.cast(-1/tf.math.square(l),dtype=tf.float32)
    w = sigma * tf.math.exp(const_2*tf.math.square(tf.math.sin(const_1*tf.math.abs(tf.math.subtract(x2,y2)))))
    return w


def SE(x,y1,params):
    l,sigma = params[0],params[1]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = 0.5*tf.cast(-1/tf.math.square(l),dtype=tf.float32)
    return sigma*tf.math.exp(tf.math.square(tf.math.subtract(y2,x2))*const_1)





