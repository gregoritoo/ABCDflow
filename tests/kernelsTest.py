import GPy
import sys
sys.path.append('../')
from kernels import PER,LIN,SE
import unittest 
import numpy as np 
import tensorflow as tf 


class KernelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(KernelTest, self).__init__(*args, **kwargs)
        self.X = tf.Variable(np.linspace(0,100,101).reshape(-1,1),dtype=tf.float64)
        self._tolerance = 1e-1


    def test_periodicKernel(self):
        lengthscale = 100*np.random.rand(1)
        period = 100*np.random.rand(1)
        variance = 100*np.random.rand(1)
        params = [lengthscale,period,variance]
        k = GPy.kern.StdPeriodic(1,lengthscale=lengthscale,period=period,variance=variance)
        assert np.allclose(PER(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    def test_periodicKernel2(self):
        params = [1,1,1]
        k = GPy.kern.StdPeriodic(1,lengthscale=1,period=1,variance=1)
        assert np.allclose(PER(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    def test_periodicKernel3(self):
        lengthscale = 100*np.random.rand(1)
        period = 100*np.random.rand(1)
        variance = 100*np.random.rand(1)
        params = [lengthscale,period,variance]
        k = GPy.kern.StdPeriodic(1,lengthscale=lengthscale,period=period,variance=variance)
        assert np.allclose(PER(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    def test_squaredExpKernel(self):
        lengthscale = 10*np.random.rand(1)
        variance = 10*np.random.rand(1)
        params = [lengthscale,variance]
        k = GPy.kern.sde_Exponential(1,lengthscale=lengthscale,variance=variance)
        assert np.allclose(SE(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    def test_squaredExpKernel2(self):
        lengthscale = 100*np.random.rand(1)
        variance = 100*np.random.rand(1)
        params = [lengthscale,variance]
        k = GPy.kern.sde_Exponential(1,lengthscale=lengthscale,variance=variance)
        assert np.allclose(SE(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    def test_squaredExpKernel3(self):
        lengthscale = 100*np.random.rand(1)
        variance = 100*np.random.rand(1)
        params = [lengthscale,variance]
        k = GPy.kern.sde_Exponential(1,lengthscale=lengthscale,variance=variance)
        assert np.allclose(SE(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    
    def test_linKernel(self):
        params = [0,1]
        k = GPy.kern.Linear(1,variances=1)
        assert np.allclose(LIN(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)

    
    def test_linKernel2(self):
        params = [0,10]
        k = GPy.kern.Linear(1,variances=10)
        assert np.allclose(LIN(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)


    def test_linKernel3(self):
        params = [0,0.5]
        k = GPy.kern.Linear(1,variances=0.5)
        assert np.allclose(LIN(self.X,self.X,tf.convert_to_tensor(params,dtype=tf.float64)).numpy(),k.K(self.X.numpy()),rtol=self._tolerance)
