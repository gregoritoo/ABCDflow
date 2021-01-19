import numpy as np 
import tensorflow as tf 
from pprint import pprint
import tensorflow_probability as tfp
import matplotlib.pyplot as plt 
import math as m
import seaborn as sn
import GPy
tf.keras.backend.set_floatx('float32')

PI = m.pi
def plot_gs(true_data,mean,X_train,X_s,stdp,stdi,color="blue"):
    plt.figure(figsize=(32,16), dpi=100)
    plt.plot(X_s,mean,color="green",label="Predicted values")
    plt.fill_between(X_s.reshape(-1,),stdp,stdi, facecolor=color, alpha=0.2,label="Conf I")
    plt.plot(X_train,true_data,color="red",label="True data")
    plt.legend()
    

def get_values(mu_s,cov_s,nb_samples=100):
    samples = np.random.multivariate_normal(mu_s,cov_s,100)
    stdp = [np.mean(samples[:,i])+1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    stdi = [np.mean(samples[:,i])-1.96*np.std(samples[:,i]) for i in range(samples.shape[1])]
    mean = [np.mean(samples[:,i])for i in range(samples.shape[1])]
    return mean,stdp,stdi
    

@tf.function
def Linear(x,y,c,sigmab,sigmav):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(tf.math.subtract(x,c*tf.ones_like(x)))
    y1 = tf.math.subtract(y,c*tf.ones_like(y))
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    w = tf.math.multiply(y2,x2)
    return sigmab+sigmav*w



@tf.function
def Periodic(x,y,l,p,sigma):
    assert x.shape[1] == y.shape[1] ,"X and Y must have the same shapes"
    _periodic = tfp.math.psd_kernels.ExpSinSquared( amplitude=sigma, length_scale=l, period=p,feature_ndims=1, name='Periodic')
    w = _periodic.apply(x,y)
    return w

@tf.function
def exp(x,y,l,sigma):
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same shapes"
    _exp = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=sigma, length_scale=l, feature_ndims=1, validate_args=False,name='ExponentiatedQuadratic')
    w = _exp.apply(x,y)
    return 


@tf.function
def compute_posterior(y,cov,cov_s,cov_ss):
    mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.01*tf.eye(cov.shape[0]))),y)
    cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),cov),cov_s) 
    return mu,cov

@tf.function
def log_l(X,Y,l,p,sigma,kernel):
    if kernel=="Periodic" :
        cov = Periodic(X,X,l,p,sigma)+0.1*tf.eye(X.shape[0])
    elif kernel == "Linear" :
        cov = Linear(X,Y,l,p,sigma)+ 0.001*tf.eye(X.shape[0])
    elif kernel =="exp":
        cov = exp(X,Y,l,p,sigma)+ 0.001*tf.eye(X.shape[0])

    loss = tf.matmul(tf.matmul(tf.transpose(Y),tf.linalg.inv(cov)),Y) + tf.math.log(tf.linalg.det(cov))+X.shape[0]*tf.math.log([PI*2])

    return loss


class PeriodicRegressor(object):
    def __init__(self):
        self._l = tf.compat.v1.get_variable('l',
                   dtype=tf.float32,
                   shape=(1,),
                   initializer=tf.random_uniform_initializer(minval=1., maxval=10.),
                   constraint=lambda z: tf.clip_by_value(z, 1, 10))
        self._p = tf.compat.v1.get_variable('p',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.),
                    constraint=lambda z: tf.clip_by_value(z, 1, 10))
        self._sigma = tf.compat.v1.get_variable('sigma',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.),
                    constraint=lambda z: tf.clip_by_value(z, 1, 10))

    @tf.function
    def __call__(self,X_train,Y_train):
        return log_l(X_train,Y_train,self._l,self._p,self._sigma,kernel="Periodic")

    @property
    def variables(self):
        return self._l,self._p,self._sigma

class LinearRegressor(object) :

    def __init__(self) :
        self._c = tf.compat.v1.get_variable('l',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._sigmab = tf.compat.v1.get_variable('p',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._sigmav = tf.compat.v1.get_variable('sigma',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
    @tf.function
    def __call__(self,X_train,Y_train):
        Linear(X_train,Y_train,self._c,self._sigmab,self._sigmav)



if __name__ =="__main__" :
    from sklearn.gaussian_process.kernels import ExpSineSquared
    X_train = tf.Variable(np.expand_dims(np.linspace((-2,2), 250), 1),dtype=tf.float32)
    Y_train = tf.Variable(np.sin(np.array(np.linspace(1,30,30)).reshape(-1, 1)),dtype=tf.float32)
    
    X_s = tf.Variable(np.arange(-2, 30, 0.2).reshape(-1, 1),dtype=tf.float32)
    mu = tf.Variable(tf.zeros((1,30)),dtype=tf.float32)

    l,p,sigma=1,1,1
    cov = Periodic(X_train,X_train,l,p,sigma)
    sn.heatmap(cov.numpy().reshape(1,-1))
    plt.show()

    """k = GPy.kern.StdPeriodic(1,l,p,sigma)
    cov2 = k.K(X_train.numpy(),X_train.numpy())
    sn.heatmap(cov2)
    plt.show()"""

    
    """learning_rate = 100
    best = 1000
    nb_restart = 1
    loop = 0
    history = {"log" : [] , "l" : [] ,"p":[]}
    while loop < nb_restart :
        model = PeriodicRegressor()
        params = {"l":l,"p":p,"sigma":sigma}
        try :
            for iteration in range(1,100):
                if iteration % 100 == 0 :
                    learning_rate = 100
                with tf.GradientTape(persistent=True) as tape :
                    print(iteration)
                    val = model(X_train,Y_train)
                print(val)
                gradient = tape.gradient(val,model.variables)
                print(gradient)
                for g,v in zip(gradient,model.variables):
                    v.assign_add(tf.constant([-learning_rate],dtype=tf.float32)*g)
        except Exception as e :
            print(e)
        if val  < best :
            best =  val
            params = {"l":model._l,"p":model._p,"sigma":model._sigma}
        loop += 1
    print(history)
    plt.plot(np.array(history["log"]))
    plt.plot(np.array(history["l"]))
    plt.plot(np.array(history["p"]))
    plt.show()
    cov = Periodic(X_train,X_train,l,p,sigma)
    cov_ss =  Periodic(X_s,X_s,l,p,sigma)
    cov_s  = Periodic(X_train,X_s,l,p,sigma)
    
    mu,cov = compute_posterior(Y_train,cov,cov_s,cov_ss)
    sn.heatmap(cov.numpy())
    plt.show()
    print(tf.linalg.det(cov).numpy())

    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=1000)
    plot_gs(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
    plt.show()
"""
