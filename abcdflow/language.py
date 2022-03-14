import numpy as np
import os
import re


def remove_useless_term_changepoint(CP):
    CP = CP.replace("CP('", "").replace("', '", ",").replace("')", "")
    CP = CP.split(",")
    CP[0] = CP[0]+":DEC_SIG"
    CP[1] = CP[1]+":INC_SIG"
    return CP


def devellopement(kernel_list):
    '''
        Devellop kernels as describe in https://github.com/duvenaud/phd-thesis 
            ex :
    inputs :
        kernel_list = list, list containing the kernels of the model
    outputs :
        splitted list of list, list containing develloped kernels
        splitted_params list of list, list containing develloped parameters associated to the kernels 
    '''
    splitted = []
    splitted_params = []
    j = 0
    first_mul = False
    for i in range(len(kernel_list)):
        if kernel_list[i][0] == "*":
            if j != i:
                splitted.append(kernel_list[j:i])
                splitted_params.append([np.arange(j, i+1)])
            for p in range(len(splitted)):
                if len(splitted[p]) > 0:
                    splitted[p].append(kernel_list[i])
                    splitted_params[p].append(j)
            j = j+1
        elif kernel_list[i][0] == "+":
            splitted.append([kernel_list[i]])
            splitted_params.append([j])
            j = j+1
        else:
            changepoints = remove_useless_term_changepoint(kernel_list[i])
            for element in changepoints:
                splitted.append([element])
            splitted_params.append([i, i+2])
            splitted_params.append([i+1, i+2])
            j = j+3
    splitted = [element for element in splitted if len(element) > 0]
    splitted_params = [
        element for element in splitted_params if element[0] != []]
    return splitted, splitted_params


def comment_changepoint(text, kern, params_dic):
    '''
        Text description for changepoints kernels 
    inputs :
        text = str, model description
        kern = str, kernel names 
        params_dic = dictionnary containing all the model parameters 
    outputs :
        text = str, model description
    '''
    if re.search('\:DEC_SIG', kern) is not None:
        text = text + \
            " which apply until  {:.1f}   ".format(
                float(params_dic["cp_x0"].numpy()[0]))
    elif re.search('\:INC_SIG', kern) is not None:
        text = text + \
            " which apply from  {:.1f}  ".format(
                float(params_dic["cp_x0"].numpy()[0]))
    return text


def comment(text, component, pos, params_dic, list_params):
    '''
        Text description using regex to extract kernels'names
    inputs :
        text = str, model description
        component = list, list of kernels 
        pos = develloped positions to extract parameters that correspond to the list of kernels 
        params_dic = dict, dictionnary containing all the model parameters 
        list_params = list, list with the name of the parameters 
    outputs :
        text = str, model description
    '''
    list_of_dic = [list_params[position] for position in pos]
    for j in range(len(component)):
        kern = component[j]
        params = list_of_dic[j]
        if re.search('\+LIN', kern) is not None:
            text = text + "\t A linear component with a offset of {:.1f} and a slope of {:.1f} , ".format(
                float(params_dic[params[0]]), float(params_dic[params[1]].numpy()[0]))
            text = comment_changepoint(text, kern, params_dic)
        if re.search('\+SE', kern) is not None:
            text = text + "\t A smooth  function with a lengthscale of {:.1f} and a variance of {:.1f} , ".format(
                float(params_dic[params[0]]), float(params_dic[params[1]].numpy()[0]))
            text = comment_changepoint(text, kern, params_dic)
        if re.search('\+PER', kern) is not None:
            text = text + "\t A periodic component with a period of {:.1f} , a variance of {:.1f} and lengthscale of {:.1f} , ".format(
                float(params_dic[params[1]].numpy()[0]), float(params_dic[params[2]].numpy()[0]), float(params_dic[params[0]].numpy()[0]))
            text = comment_changepoint(text, kern, params_dic)
        if re.search('\+CONST', kern) is not None:
            text = text + "\t A constant component with a offset of {:.1f} , ".format(
                float(params_dic[params[0]].numpy()[0]))
            text = comment_changepoint(text, kern, params_dic)
        if re.search('\*LIN', kern) is not None:
            if re.search('\*LIN\*LIN', kern) is not None:
                text = text + "with polynomial varying amplitude  of  {:.1f} , ".format(
                    float(params_dic[params[0]].numpy()[0]))
            else:
                text = text + "with a linearely varying amplitude  of {:.1f} , ".format(
                    float(params_dic[params[0]].numpy()[0]))
        if re.search('\*PER', kern) is not None:
            text = text + "modulated by a periodic function defined by a period of {:.1f}, a variance of {:.1f} and lengthscale of {:.1f} , ".format(
                float(params_dic[params[1]].numpy()[0]), float(params_dic[params[2]].numpy()[0]), float(params_dic[params[0]].numpy()[0]))
        if re.search('\*SE', kern) is not None:
            text = text + "whose shape varying smoothly with a lengthscale of {:.1f} , ".format(
                float(params_dic[params[0]].numpy()[0]))
    return text[:-2]+"."


def comment_gpy(text, component, pos, variables_names, variables):
    list_of_dic = [variables[position] for position in pos]
    for j in range(len(component)):
        kern = component[j]
        params = list_of_dic[j]
        if re.search('\+LIN', kern) is not None:
            text = text + \
                "\t A linear component with  a slope of {:.1f} , ".format(
                    params[0])
        if re.search('\+SE', kern) is not None:
            text = text + \
                "\t A smooth  function with a lengthscale of {:.1f} and a variance of {:.1f} , ".format(
                    params[1], params[0])
        if re.search('\+PER', kern) is not None:
            text = text + "\t A periodic component with a period of {:.1f}, a variance of {:.1f} and lengthscale of {:.1f} , ".format(
                params[1], params[0], params[2])
        if re.search('\*LIN', kern) is not None:
            if re.search('\*LIN\*LIN', kern) is not None:
                text = text + \
                    "with polynomial varying amplitude  of  {:.1f} , ".format(
                        params[0])
            else:
                text = text + \
                    "with a linearely varying amplitude  of {:.1f} , ".format(
                        params[0])
        if re.search('\*PER', kern) is not None:
            text = text + "modulated by a periodic function defined by a period of {:.1f}, a variance of {:.1f} and lengthscale of {:.1f} , ".format(
                params[1], params[0], params[2])
        if re.search('\*SE', kern) is not None:
            text = text + \
                "whose shape varying smoothly with a lengthscale of {:.1f} , ".format(
                    params[1], params[0])
    return text[:-2]+"."
