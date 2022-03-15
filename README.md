# ABCDflow 

Unofficial implementation of https://arxiv.org/pdf/1302.4922.pdf and https://github.com/duvenaud/phd-thesis


Medium article : https://medium.com/analytics-vidhya/automatic-time-series-description-using-gaussian-processes-b9bf09f6d0ce

This project uses three differents tehniques to maximize the log likelihood :
- Gradient descent using tensorflow 
- L-BFGS-B optimizer (scipy)
- Wrapper of GPy library

How to use it ?
==============

Install dependence libraries 

```
    pip install -r requirements.txt
```

Using tensorflow optimization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from abcdflow.trainer import Trainer

if __name__ =="__main__" :
    #load data
    Y = np.array(pd.read_csv("./data/co2.csv")["x"][:600]).reshape(-1, 1)
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    # points to predict
    X_s = np.linspace(0,len(Y)+30,len(Y)+30).reshape(-1, 1)
    
    #launch search of best kernels using scipy optimizer and multithreading with 10 random restart for each optimization step
    training_class = Trainer(X, Y, X_s, straigth=True, GPY=False, do_plot=True, depth=2, verbose=True, initialisation_restart=10, reduce_data=False,
                             experimental_multiprocessing=True, use_changepoint=True, base_kernels=["+PER", "+LIN", "+SE"])
    
    model, kernel = training_class.launch_analysis()
    
    #Textual description
    model.describe(kernel)
    
    #predict posterior mean and covariance
    mu,cov = model.predict(X,Y,X_s,kernel)
    
    # Plot results 
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()
    
    #Plot contribution of every group of kernel using kernels devellopement as in the article 
    model.decompose(kernel,X,Y,X_s)
    plt.show()
```

Using wrapper of GPy

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from abcdflow.trainer import Trainer

if __name__ =="__main__" :
    #load data
    Y = np.array(pd.read_csv("./data/co2.csv")["x"][:600]).reshape(-1, 1)
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    # points to predict
    X_s = np.linspace(0,len(Y)+30,len(Y)+30).reshape(-1, 1)
    
    #launch search of best kernels using scipy optimizer and multithreading with 10 random restart for each optimization step
    training_class = Trainer(X, Y, X_s, straigth=True, GPY=True, do_plot=True, depth=2, verbose=True, initialisation_restart=10, reduce_data=False,
                             experimental_multiprocessing=True, use_changepoint=True, base_kernels=["+PER", "+LIN", "+SE"])
    
    model, kernel = training_class.launch_analysis()
    
    #Textual description
    model.describe(kernel)
    
    #predict posterior mean and covariance
    mu,cov = model.predict(X,Y,X_s,kernel)
    
    # Plot results 
    model.plot(mu,cov,X,Y,X_s,kernel)
    plt.show()
    
```
> :warning: **The wrapper is still a work in progress**: The decomposition method as well as the changepoints aren't implemented for the GPy's wrapper 


Results ?
=======
Prediction
----------
![Prediction](imgs/final_final.png)

Description
-----------
![Description](imgs/description.png)

Decomposition
-------------
![Decompo](imgs/decomp.png)
