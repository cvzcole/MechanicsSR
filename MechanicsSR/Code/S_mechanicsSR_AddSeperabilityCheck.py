from __future__ import print_function
import torch
import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from itertools import combinations
import time
import math

def check_separability_plus(pathdir, filename):
    print('start to check Additional separability')
    # separability tolerance (hyperparameter)
    tolerance = 0.05
    # compare first 4 digits
    digits_tolerance = 1e-3
    # load the data
    n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
    variables = np.loadtxt(pathdir+filename, usecols=(0,))
    ogdata = np.loadtxt(pathdir+filename)

    if n_variables==1:
        print(filename, "just one variable for ADD")
        # if there is just one variable you have nothing to separate
        return (-1,-1,-1)
    else:
        for j in range(1,n_variables):
            v = np.loadtxt(pathdir+filename, usecols=(j,))
            variables = np.column_stack((variables,v))

    f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
    f_dependent = np.reshape(f_dependent,(len(f_dependent),1))   
    factors = torch.from_numpy(variables) 
    factors = factors.float()

    product = torch.from_numpy(f_dependent)
    product = product.float()

    fact_vary = factors.clone()
    for k in range(len(factors[0])):
        fact_vary[:,k] = torch.full((len(factors),),torch.median(factors[:,k]))
        fact_mean = fact_vary.numpy()
        fact_mean = np.unique(fact_mean, axis=0) # find c1 c2

    #print('mean_value was found in dataset=', fact_mean)
    matching_rows = np.where(np.all(np.isclose(variables, fact_mean, atol=digits_tolerance),axis=1))[0]
    row_idx = matching_rows[0]
    fact_constant = ogdata[row_idx,-1]
    print('constant=',fact_constant)
    #print('variables',variables)
    #print('row_idx=, function_constant = ', row_idx,fact_constant)
    # loop through all indices combinations
    var_indices_list = np.arange(0,n_variables,1)
    min_error = 1000
    best_i = []
    best_j = []

    for i in range(1,n_variables):
        c = combinations(var_indices_list, i)
        for j in c:
            print('Now check ith variable_add , idx = ',j)
            x_bary_idx=[]
            x_bary_result=[]
            fact_vary_one = factors.clone()
            fact_vary_rest = factors.clone()
            rest_indx = list(filter(lambda x: x not in j, var_indices_list))
            for t1 in rest_indx:
                xy_bar_result=[]
                fact_vary_one[:,t1] = torch.full((len(factors),),torch.median(factors[:,t1]))
                fact_one = fact_vary_one.numpy()
                #print('fact_one',fact_one)
                fact_one = np.unique(fact_one, axis=0)
                #print('fact_one',fact_one)  # fact_one is {x,ybar} without repeat x, which is correct.
                i_row_max = fact_one.shape[0] 
                i_variables_max = variables.shape[0]
                #print('variables=',variables[2,:])
                #print('i_row_max=',i_row_max)  # i_row_max is the number of rows in {x,ybar}
                for i_row in range(i_row_max):
                    matching_rows_one = np.where(np.isclose(variables, fact_one[i_row,:], atol = digits_tolerance))[0][0]
                    #print('i_row=',i_row)
                    #print('fact_i',fact_one[i_row,:])
                    #print('len_fact_one',fact_one.shape[1])
                    for i_variables in range(i_variables_max):
                        if sum(np.isclose(variables[i_variables,:],fact_one[i_row,:],atol = digits_tolerance)) == fact_one.shape[1]:
                            #print('isclose_result = ', np.isclose(variables[i_variables,:],fact_one[i_row,:],atol = digits_tolerance))
                            #print('i_variables=', i_variables)
                            xy_bar_result=  np.append(xy_bar_result,ogdata[i_variables,-1])
                            break
                    #print('xy_bar_result=', xy_bar_result) 
                xy_bar_result = np.transpose(xy_bar_result) 
                xy_bar = np.column_stack((fact_one[:,j],xy_bar_result))    
                #print('xy_bar',xy_bar)

            for t2 in j:
                x_bary_result=[]
                fact_vary_rest[:,t2] = torch.full((len(factors),),torch.median(factors[:,t2]))
                fact_rest = fact_vary_rest.numpy()
                fact_rest = np.unique(fact_rest, axis=0)
                i_row_max = fact_rest.shape[0]
                for i_row in range(i_row_max):
                    for i_variables in range(i_variables_max):
                        if sum(np.isclose(variables[i_variables,:],fact_rest[i_row,:],atol = digits_tolerance)) == fact_rest.shape[1]:
                            #print('isclose_result = ', np.isclose(variables[i_variables,:],fact_rest[i_row,:],atol = digits_tolerance))
                            #print('i_variables=', i_variables)
                            x_bary_result=  np.append(x_bary_result,ogdata[i_variables,-1])
                            break
                    matching_rows_rest = np.where(np.isclose(variables, fact_rest[i_row,:], atol= digits_tolerance))[0][0]
                    x_bary_idx= np.append(x_bary_idx,matching_rows_rest)
                x_bary_result = np.transpose(x_bary_result)
                x_bary = np.column_stack((fact_rest[:,rest_indx],x_bary_result))
                #print('x_bary=',x_bary)        

            # Formula ogdata[:,-1] - xy_bar - xbar_y + fact_constant
            Er_add = 0
            xy_bar_search = fact_one[:,j]
            x_bary_search = fact_rest[:,rest_indx]
            #print(xy_bar_search)
            idx_ogdata = ogdata.shape[0]
            idx_ybar = xy_bar.shape[0]
            idx_xbar = x_bary.shape[0]
            
            #initiate the idx_number of matching point
            er_idx = 0
            for t3 in range(idx_ogdata):
                t3_found=0
                for t_x in range(idx_ybar):
                    if np.all(ogdata[t3,j]==xy_bar_search[t_x]):
                        for t_y in range(idx_xbar):
                            if np.all(ogdata[t3,rest_indx]== x_bary_search[t_y]):
                                # find a new matching point idx+1
                                er_idx = er_idx+1
                                section2 = - xy_bar[t_x,-1] - x_bary[t_y,-1]
                                new_er = abs( ogdata[t3,-1] + section2 + fact_constant)
                                new_er = abs(new_er / ogdata[t3,-1])
                                # Check each part of this formula:
                                Er_add = Er_add + new_er
                                t3_found = 1
                                break
                    if t3_found ==1:
                        break
            if er_idx > 0:
                print('er_idx=',er_idx)
                mse= Er_add / er_idx
                                # if error is above tolerance then quit
            if mse>tolerance:
                print('no additive separability Er_total=',mse)
                return 99999,j,rest_indx

            if mse<=tolerance:
                print('addittive separability found, er=',mse)
                print('variable_idx = ', j)
                min_error = mse
                best_i = j
                best_j = rest_indx
                    
    return min_error, best_i, best_j