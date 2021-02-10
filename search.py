
import numpy as np  
import os
import sys 
import pandas as pd 
from utils import KERNELS_LENGTH,KERNELS,KERNELS_OPS,GPY_KERNELS
import itertools 
borne = -1*10e40



def mulkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a * 
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex *PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER*PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","*PER"]
    '''
    kernels_name = "("+kernels_name+")" + "*"+new_k
    _kernel_list = _kernel_list + ["*"+new_k]
    return kernels_name,_kernel_list



def addkernel(kernels_name,_kernel_list,new_k):
    ''' 
        Add  new kernel to the names and in the kernel list with a +
    inputs :
        kernels_name :  string, name of the full kernel  ex +LIN*PER
        _kernels_list : list of string, list of kernels with theirs according operation ex ["+LIN","*PER"]
        new_k : string, name of kernel to add with it according operation ex +PER
    outputs :
        kernels_name :  string, updated kernel name   ex +LIN*PER+PER
        _kernels_list : list of string, list of updated kernels ex ["+LIN","*PER","+PER"]
    '''
    kernels_name = kernels_name + "+"+new_k
    _kernel_list = _kernel_list + ["+"+new_k]
    return kernels_name,_kernel_list

def preparekernel(_kernel_list):
    '''
        Receive the list of kernels with theirs operations and return a dict with the kernels names and parameters 
    inputs :
        _kernels_list : list of string, list of kernels with theirs according operation
    outputs :
        kernels : dict, dict cotaining kernel parameters (see KERNELS)
    '''
    dic = tuple([KERNELS[d[1:]] for d in _kernel_list])
    kernels={}
    i = 1
    for para in dic :
        for key,value in para.items() :
            if key in kernels.keys() :
                if key != "noise" :
                    key = key+"_"+str(i)
                    kernels.update({key:[element+"_"+str(i) for element in value]})
            else :
                kernels.update({key:[element for element in value]})
            i+=1
    return kernels



def search(kernels_name,_kernel_list,init,depth=5):
    '''
        Return all the possible combinaison of kernels starting with a '+' 
    inputs :
        kernels_name :  string, not used 
        _kernel_list : list of tuples, not used 
        init : Bool, not used 
    '''
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    for i in range(1,depth) :
        if i == 1 : combination =  list(itertools.combinations(kerns, i))
        else : combination = list(itertools.permutations(kerns, i)) 
        for comb in combination :
            if not comb[0][0] == "*" : 
                COMB.append(comb)
    return COMB


def prune(tempbest,rest):
    '''
        Cut the graph in order to keep only the combinaison that correspond to the best for the moment 
    inputs :
        tempbest : list ,  names of best combinaisons of kernels already trained 
        rest : list of tuples , all the combinaisons of kernels not trained yet 
    outputs :
        new_rest : list of tuples , all the combinaisons to be trained 
    '''
    print("Prunning {} elements".format(len(rest)))
    new_rest = []
    for _element in rest :
        for _best in tempbest :
            _element = list(_element)
            if _best == _element[:len((_best))] and _element not in new_rest :
                new_rest.append(_element)
    print("Prunning ended, still {} kernels to try".format(len(new_rest)))
    return new_rest



def search_and_add(kernel_tuple):
    ''' 
        Return all possible combinaison for one step
    inputs :
        _kernels_list : list, not used 
    '''
    kerns = tuple((KERNELS_OPS.keys()))
    COMB = []
    combination =  list(itertools.combinations(kerns, 1))
    for comb in combination :
        COMB.append(kernel_tuple+comb)
    return COMB


def replacekernel(_kernel_list):
    COMB = []
    counter=0
    replaced = list(_kernel_list)
    try :
        for element in _kernel_list :
            for replace_object in KERNELS.keys() :
                if element[1:] != replace_object :
                    replaced[counter] = element[0] + replace_object
                if replaced not in COMB : COMB.append(replaced)
                replaced = list(_kernel_list)
            counter +=1
        return COMB
    except Exception as e :
        print(e)
        return _kernel_list

def gpy_kernels_from_names(_kernel_list):
    kernel = GPY_KERNELS[_kernel_list[0][1 :]]
    for j in range(1,len(_kernel_list)) :
        if _kernel_list[j][0] == "+" :
            kernel = kernel + GPY_KERNELS[_kernel_list[j][1 :]]
        elif _kernel_list[j][0] == "*" :
            kernel = kernel * GPY_KERNELS[_kernel_list[j][1 :]]
        else :
            raise ValueError("Illicite operation")
    return kernel

