
import numpy as np 
import pandas as pd 
import bocd
import ruptures as rpt
from pyinform.blockentropy import block_entropy


def cut_signal(signal):

    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(300), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))
    rt_mle = np.empty(signal.shape)
    for i, d in enumerate(signal):
        bc.update(d)
        rt_mle[i] = bc.rt
    index_changes = np.where(np.diff(rt_mle)<0)[0]
    return index_changes


def changepoint_detection(ts,percent=0.05,plot=True,num_c=4):
    length = len(ts)
    bar = int(percent*length)
    ts = np.array(ts) [bar:-bar]
    min_val,model = length, "l1" 
    algo = rpt.Dynp(model="normal").fit(np.array(ts))
    dic = {"best":[0,length]}
    try :
        for i in range(num_c) :
            my_bkps = algo.predict(n_bkps=i)
            if plot :
                rpt.show.display(np.array(ts), my_bkps, figsize=(10, 6))
                plt.show()
            start_borne,full_entro = 0,0
            for end_borne in my_bkps :
                val = block_entropy(ts[start_borne:end_borne], k=1)   
                full_entro = val + full_entro
                start_borne = end_borne
            if full_entro == 0 : break
            elif full_entro < min_val :
                min_val = full_entro
                dic["best"] = [0]+my_bkps
            else : pass 
    except Exception as e :
        print(e)
        print("Not enough point")
        return {"best":[0,length]}
    return dic