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
def Periodic(x,y1,l,p,sigma):
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

@tf.function
def exp(x,y1,l,sigma):
    assert x.shape[1] == y1.shape[1] ,"X and Y must have the same shapes"
    x1 = tf.transpose(x)
    multiply_y = tf.constant([1,x.shape[0]])
    y2 = tf.transpose(tf.tile(y1, multiply_y))
    multiply_x = tf.constant([y1.shape[0],1])
    x2 = tf.transpose(tf.tile(x1, multiply_x))
    const_1 = 0.5*tf.cast(-1/tf.math.square(l),dtype=tf.float32)
    return sigma*tf.math.exp(tf.math.square(tf.math.subtract(y2,x2))*const_1)


@tf.function
def compute_posterior(y,cov,cov_s,cov_ss):
    mu = tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.01*tf.eye(cov.shape[0]))),y)
    cov = cov_ss - tf.matmul(tf.matmul(tf.transpose(cov_s),tf.linalg.inv(cov+0.01*tf.eye(cov.shape[0]))),cov_s)
    return mu,cov

@tf.function
def log_l(X,Y,params,kernel):
    if kernel=="Periodic" :
        cov = Periodic(X,Y,l=params["l"],p=params["p"],sigma=params["sigma"])+1*tf.eye(X.shape[0])
    elif kernel == "Linear" :
        cov = Linear(X,Y,c=params["c"],sigmab=params["sigmab"],sigmav=params["sigmav"])+1*tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,Y,l=params["l"],sigma=params["sigma"])+ 0.001*tf.eye(X.shape[0])
    loss = -0.5*tf.matmul(tf.matmul(tf.transpose(Y),tf.linalg.inv(cov)),Y) - 0.5*tf.math.log(tf.linalg.det(cov))-0.5*X.shape[0]*tf.math.log([PI*2])
    
    return -loss

@tf.function
def log_cholesky_l(X,Y,params,kernel):
    if kernel=="Periodic" :
        cov = Periodic(X,X,l=params["l"],p=params["p"],sigma=params["sigma"])+1*tf.eye(X.shape[0])
    elif kernel == "Linear" :
        cov = Linear(X,X,c=params["c"],sigmab=params["sigmab"],sigmav=params["sigmav"])+ tf.eye(X.shape[0])
    elif kernel =="SE":
        cov = exp(X,X,l=params["l"],sigma=params["sigma"]) + tf.eye(X.shape[0])
    _L = tf.linalg.cholesky(cov)
    _temp = tf.linalg.solve(_L, Y)
    alpha = tf.linalg.solve(tf.transpose(_L), _temp)
    """ _temp = tf.matmul(tf.linalg.inv(_L),Y)
    LT = tf.transpose(tf.linalg.inv(_L))
    alpha = tf.matmul(LT,_temp)"""
    loss = 0.5*tf.matmul(tf.transpose(Y),alpha) + tf.math.log(tf.linalg.trace(_L)) +0.5*X.shape[0]*tf.math.log([PI*2])
    return loss


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
        return self._l,self._p,self._sigma#,self._noise

class LinearRegressor(object) :

    def __init__(self) :
        self._c = tf.compat.v1.get_variable('l',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=100.))
                   
        self._sigmab = tf.compat.v1.get_variable('sigmab',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=100.))
        self._sigmav = tf.compat.v1.get_variable('sigmav',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=0.01, maxval=1.))
    @tf.function
    def __call__(self,X_train,Y_train):
        
        params={"c":self._c,"sigmab":self._sigmab,"sigmav":self._sigmav}
        return log_cholesky_l(X_train,Y_train,params,kernel="Linear")

    @property
    def variables(self):
        return self._c,self._sigmab,self._sigmav


class SquaredExpRegressor(object) :

    def __init__(self) :
        self._l = tf.compat.v1.get_variable('l',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
        self._sigma = tf.compat.v1.get_variable('sigmab',
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.random_uniform_initializer(minval=1., maxval=10.))
                    

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




if __name__ =="__main__" :
    X_train = tf.Variable(np.array(np.linspace(1,60,30)).reshape(-1, 1),dtype=tf.float32)
    Y_train = tf.Variable(np.sin(np.array(np.linspace(1,60,30)).reshape(-1, 1)),dtype=tf.float32)
    
    X_s = tf.Variable(np.arange(-2, 70, 0.5).reshape(-1, 1),dtype=tf.float32)
    mu = tf.Variable(tf.zeros((1,30)),dtype=tf.float32)

    
    """l,p,sigma=1,2,54

    model = PeriodicRegressor()

  
    params={"l":l,"p":p,"sigma":sigma,"noise":0.001}

    cov = Periodic(X_train,X_train,l,p,sigma)
    k = GPy.kern.StdPeriodic(lengthscale=l, input_dim=1, variance=sigma,period=p)
    cov2 = k.K(X_train.numpy(),X_train.numpy())

    sn.heatmap(cov2-cov.numpy())
    plt.show()


    


    print(log_cholesky_l(X_train,Y_train,params,"Periodic"))
    

    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
    m.Gaussian_noise = 1
    print(m)
    print(m.log_likelihood())


 

    print("je") 
    """
    optimizer = tf.optimizers.Adamax(learning_rate=.01)
    l,p,sigma=1,1,1
    best = 10e40
    nb_restart = 5
    loop = 0
    nb_epochs = 1000
    history = {"log" : [] ,"p":[],"l":[],"sigma":[],"erro":[]}
    while loop < nb_restart :
        model = SquaredExpRegressor()
        try :
            for iteration in range(1,nb_epochs):
                print(iteration)
                if iteration % 100 == 0 :
                    learning_rate = 0.1
                with tf.GradientTape(persistent=False) as tape :
                    tape.watch(model.variables)
                    val = model(X_train,Y_train)
                    """k = GPy.kern.StdPeriodic(lengthscale=model._l, input_dim=1, variance=model._sigma,period=model._p)
                    m = GPy.models.GPRegression(X_train.numpy(), Y_train.numpy(), k, normalizer=False)
                    m.Gaussian_noise = 1"""
                history["log"].append(val.numpy())
                history["l"].append(model._l.numpy())
                history["sigma"].append(model._sigma.numpy())
                gradient = tape.gradient(val,model.variables)
                optimizer.apply_gradients(zip(gradient, model.variables))
                sys.stdout.write("\r"+"="*int(iteration/nb_epochs*50)+">"+"."*int((nb_epochs-iteration)/nb_epochs*50)+"|"+" * log likelihood  is : {} at epoch : {} ".format(val,iteration))
                sys.stdout.flush()
        except Exception as e :
            print(e)
        if val  < best :
            best =  val
            best_model = model
        loop += 1

    k = GPy.kern.StdPeriodic(lengthscale=l, input_dim=1, variance=sigma)       
    m = GPy.models.GPRegression(X_train.numpy(), X_train.numpy(), k, normalizer=False)
    m.Gaussian_noise = 1
    m.optimize()
    print(m)

    print(best_model.variables)

    #plt.plot(np.array(history["log"]).reshape(-1,1))
    plt.plot(np.array(history["p"]).reshape(-1,1),label="p")
    plt.plot(np.array(history["log"]).reshape(-1,1),label="l")
    plt.plot(np.array(history["sigma"]).reshape(-1,1),label="sigma")
    #plt.plot(np.array(history["erro"]).reshape(-1,1),label="erro")
    plt.legend()
    plt.show()
    mu,cov = best_model.predict(X_train,Y_train,X_s)    
    sn.heatmap(cov.numpy())
    plt.show()

    mean,stdp,stdi=get_values(mu.numpy().reshape(-1,),cov.numpy(),nb_samples=1000)
    plot_gs(Y_train.numpy(),mean,X_train.numpy(),X_s.numpy(),stdp,stdi)
    plt.show()