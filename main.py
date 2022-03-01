import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 

from abcdflow.training import launch_analysis






if __name__ =="__main__" :
    #Y = np.append(np.linspace(1,200,200),200+10*np.sin(np.linspace(1,200,200))).reshape(-1, 1)
    Y = np.array(pd.read_csv("./data/co2.csv")["x"][:600]).reshape(-1,1)
    #Y = np.append(5+40*np.sin(0.25*np.linspace(0, 100, 101)),2*np.linspace(5, 250, 246)).reshape(-1, 1)
    #plt.plot(Y)
    #plt.show()
    X_s = np.linspace(0,len(Y)+50,len(Y)+50).reshape(-1, 1)
    X = np.linspace(0,len(Y),len(Y)).reshape(-1,1)
    #plt.plot(Y)
    #plt.show()
    t0 = time.time()
    """X = np.linspace(-10, 10, 101)[:, None].reshape(-1, 1)
    Y = np.array(np.cos( (X - 5) / 2 )**2 * X * 2).reshape(-1, 1)
    X_s = np.linspace(-20,20,len(Y)+20).reshape(-1, 1)
    plt.plot(Y)
    plt.show()"""
    #X_s = np.linspace(-20,20,len(X)+40).reshape(-1, 1)
    t0 = time.time()
    model,kernel= launch_analysis(X,Y,X_s,straigth=True,GPY=True,do_plot=True,depth=1,verbose=True,initialisation_restart=9,reduce_data=False,experimental_multiprocessing=True,use_changepoint=False,base_kernels=["+PER","+LIN","+SE"]) #straight parameters == True
    print(model,kernel)
    print('time took: {} seconds'.format(time.time()-t0))
    model.describe(kernel)
    mu,cov = model.predict(X,Y,X_s,kernel)
    model.plot(mu,cov,X,Y,X_s,kernel)
    model.plot()
    plt.show()
    #model.decompose(kernel,X,Y,X_s)

