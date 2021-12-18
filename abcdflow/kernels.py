
import logging
import math as m
import sys 
import os 

import numpy as np 
import tensorflow as tf 

logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.keras.backend.set_floatx('float64')

PI = m.pi
_precision = tf.float64

@tf.function
def LIN(x,y,params):
    '''
        Linear kernel : LIN(x,x') = sigma²*(x-c)*(x'-c)
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    c,sigmav = params[0],params[1]
    x1 = tf.transpose(tf.math.subtract(x,c*tf.ones_like(x,dtype=_precision)))
    y1 = tf.math.subtract(y,c*tf.ones_like(y,dtype=_precision))
    y2 = tf.transpose(tf.tile(y1, tf.constant([1,x.shape[0]])))
    x2 = tf.transpose(tf.tile(x1, tf.constant([y.shape[0],1])))
    return tf.math.square(sigmav)*tf.math.multiply(y2,x2) 

@tf.function
def CONST(x,y,sigma):
    '''
        Constant kernel : CONST(x,x') = sigma
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    return sigma*tf.ones_like(x2)

@tf.function
def WN(x,y,sigma):
    '''
        White noise kernel : WN(x,x') = sigma if x=x'
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    x2 = tf.transpose(tf.tile(x1, tf.constant([y.shape[0],1])))
    y2 = tf.transpose(tf.tile(y, tf.constant([1,x.shape[0]])))
    w = tf.cast(tf.math.logical_not(tf.not_equal( tf.math.subtract(x2,y2), tf.constant(0, dtype=_precision)), name="inverse"),dtype=_precision)
    return sigma*w

@tf.function
def PER(x,y1,params):
    '''
        Periodic kernel : PER(x,x') = sigma*exp(-0.5/l²*sin(pi/p*abs(x-x')²))
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    l,p,sigma = params[0],params[1],params[2]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = tf.cast(PI/p,dtype=_precision)
    const_2 = tf.cast(2*(-1/tf.math.square(l)),dtype=_precision)
    w = sigma * tf.math.exp(const_2*tf.math.square(tf.math.sin(const_1*tf.math.abs(tf.math.subtract(x2,y2)))))
    return w

@tf.function
def SE(x,y1,params):
    '''
        Square exponential (RBF) kernel : SE(x,x') = sigma * exp(-0.5/l²*(x-x')²)
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    l,sigma = params[0],params[1]
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 =  tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 =  tf.cast(tf.transpose(tf.tile(x1, multiply_x)),dtype=_precision)
    const_1 = tf.cast(0.5*tf.cast(-1/tf.math.square(l),dtype=_precision),dtype=_precision)
    return sigma*tf.math.exp(tf.math.square(tf.math.subtract(y2,x2))*const_1)



@tf.function
def RQ(x,y,params):
    '''
        Rational quadratic kernel : RQ(x,x') = σ(1+(x−x′)^2*αℓ2)^−α
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
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


"""
    sigmoid and dec_sigmoid 
"""
def sigmoid(x,s,cp):
    temp = cp*tf.ones_like(x,dtype=tf.float64)
    return tf.math.divide(tf.ones_like(x,dtype=tf.float64),tf.math.add(tf.ones_like(x,dtype=tf.float64),tf.math.exp(-s*tf.math.subtract(x,temp))))

def dec_sigmoid(y,s,cp):
    one = tf.ones_like(y,dtype=tf.float64)
    temp = cp*tf.ones_like(y,dtype=tf.float64)
    return tf.math.subtract(one,tf.math.divide(tf.ones_like(y,dtype=tf.float64),tf.math.add(tf.ones_like(y,dtype=tf.float64),tf.math.exp(-s*tf.math.subtract(y,temp)))))

def tanh(x,s,cp):
    temp = cp*tf.ones_like(x,dtype=tf.float64)
    return 0.5*tf.math.add(tf.ones_like(x,dtype=tf.float64),1/s*tf.math.subtract(temp,x))

def dec_tanh(x,s,cp):
    one = tf.ones_like(x,dtype=tf.float64)
    temp = cp*tf.ones_like(x,dtype=tf.float64)
    return tf.math.subtract(one,0.5*tf.math.add(tf.ones_like(x,dtype=tf.float64),1/s*tf.math.subtract(temp,x)))








def CP(x,y,params,left_kern,rigth_kern,left_params,rigth_params):
    '''
        Changepoint kernel 
    inputs :
        x : tensor, X_train
        y : tensor, Y_train
    outputs :
        covariance tensor
    '''
    cp,s  = params[1],params[0]
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same dimension"
    x1 = tf.transpose(x)
    x2 = tf.transpose(tf.tile(x1, tf.constant([y.shape[0],1])))
    y2 = tf.transpose(tf.tile(y, tf.constant([1,x.shape[0]])))


    sig_y,sig_x = sigmoid(x2,s,cp),sigmoid(y2,s,cp)
    neg_sig_y,neg_sig_x = dec_sigmoid(y2,s,cp),dec_sigmoid(x2,s,cp)

    left = tf.math.multiply(tf.math.multiply(neg_sig_y,neg_sig_x),left_kern(x,y,left_params))
    rigth = tf.math.multiply(tf.math.multiply(sig_y,sig_x),rigth_kern(x,y,rigth_params))
    return tf.math.add(left,rigth)