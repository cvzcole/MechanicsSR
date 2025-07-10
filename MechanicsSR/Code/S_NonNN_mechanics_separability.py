from __future__ import print_function
from decimal import Decimal, getcontext
import torch
import os
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
from itertools import combinations
import time

is_cuda = torch.cuda.is_available()

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

is_cuda = torch.cuda.is_available()

def do_separability_plus(pathdir, filename, list_i,list_j):
    np.set_printoptions(precision=20, suppress=False)
    try:
        # load the data
        fullpath = pathdir + filename
        n_variables = np.loadtxt(fullpath, dtype='str').shape[1] - 1
        print("number of variables =", n_variables)
        
        variables = np.loadtxt(fullpath, usecols = (0,))

        if n_variables==1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1, n_variables):
                v = np.loadtxt(fullpath, usecols = (j,))
                variables = np.column_stack((variables, v))
            for j in range(1, n_variables + 1):
                v = np.loadtxt(fullpath, usecols = (j,))
                ogdata = np.column_stack((variables, v))

        print('check here ogdata', variables)

        f_dependent = np.loadtxt(fullpath, usecols = (n_variables,))
        f_dependent = np.reshape(f_dependent, (len(f_dependent), 1))

        factors = variables.astype(np.double) 
        #print(np.array2string(factors, formatter={'all': lambda x: f"{x:.20f}"}))

        product = f_dependent.astype(np.double)
        #print(np.array2string(product, formatter={'all': lambda x: f"{x:.20f}"}))
        
        fact_vary_one = np.copy(factors)
        fact_vary_rest = np.copy(factors)
       
        for t1 in list_j:
            fact_vary_one[:, t1] = np.median(factors[:, t1])
        for t2 in list_i:
            fact_vary_rest[:, t2] = np.median(factors[:, t2])
        # Prepare outputs
        data_sep_1 = np.empty((0, n_variables + 1))
        data_sep_2 = np.empty((0, n_variables + 1))

        # Save first part
        for i in range(len(fact_vary_one)):
            ck1 = fact_vary_one[i]
            for j in range(len(factors)):
                ck2 = factors[j]
                if np.all(ck1 == ck2):
                    new_row = ogdata[j, :]
                    data_sep_1 = np.vstack([data_sep_1, new_row])
                    break
        
        data_sep_1 = np.delete(data_sep_1, list_j, axis=1)
        #data_sep_1 = np.array([[float(x) for x in row] for row in data_sep_1])
        
        print('additive datasep1 prepared', data_sep_1)

        # Save second part
        for i in range(len(fact_vary_rest)):
            ck1 = fact_vary_rest[i]
            for j in range(len(factors)):
                ck2 = factors[j]
                if np.all(ck1 == ck2):
                    new_row = ogdata[j, :]
                    data_sep_2 = np.vstack([data_sep_2, new_row])
                    break
       
        data_sep_2 = np.delete(data_sep_2, list_i, axis=1)
        #data_sep_2 = np.array([[np.double(x) for x in row] for row in data_sep_2])
       
        print('additive datasep2 prepared', data_sep_2)

        try:
            os.makedirs("results/separable_add/", exist_ok=True)
        except:
            pass
        
        str1 = filename + "-add_a"
        str2 = filename + "-add_b"
        str3 = filename + "-og"

        np.savetxt("results/separable_add/" + str1, np.round(data_sep_1, 5), fmt="%.5f", delimiter=" ", newline="\n", encoding="utf-8")
        np.savetxt("results/separable_add/" + str2, np.round(data_sep_2, 5), fmt="%.5f", delimiter=" ", newline="\n", encoding="utf-8")

        return ("results/separable_add/", str1, "results/separable_add/", str2)
    except Exception as e:
        print("Error:", e)
        return (-1, -1)

def do_separability_multiply(pathdir, filename, list_i, list_j):
    try:
        # load the data
        fullpath = pathdir + filename
        n_variables = np.loadtxt(fullpath, dtype='str').shape[1] - 1
        variables = np.loadtxt(fullpath, usecols = (0,))

        if n_variables==1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1, n_variables):
                v = np.loadtxt(fullpath, usecols = (j,))
                variables = np.column_stack((variables, v))
            for j in range(1, n_variables + 1):
                v = np.loadtxt(fullpath, usecols = (j,))
                ogdata = np.column_stack((variables, v))

        print('check here ogdata', np.array([[float(x) if isinstance(x, Decimal) else x for x in row] for row in variables]))

        f_dependent = np.loadtxt(fullpath, usecols = (n_variables,))
        f_dependent = f_dependent.reshape((len(f_dependent), 1))

        factors = variables.astype(np.double)
        product = f_dependent.astype(np.double)

        # Create modified versions of the factors
        fact_vary_one = np.copy(factors)
        fact_vary_rest = np.copy(factors)

        for t1 in list_j:
            fact_vary_one[:, t1] = np.median(factors[:, t1])
        for t2 in list_i:
            fact_vary_rest[:, t2] = np.median(factors[:, t2])

        print('constants here', [np.median(factors[:, t]) for t in list_j + list_i])

        # Initialize empty result arrays
        data_sep_1 = np.empty((0, n_variables + 1))
        data_sep_2 = np.empty((0, n_variables + 1))

        # Match fact_vary_one rows with original factors
        for i in range(len(fact_vary_one)):
            ck1 = fact_vary_one[i]
            for j in range(len(factors)):
                ck2 = factors[j]
                if np.allclose(ck1, ck2, rtol=1e-8, atol=1e-8):
                    new_row = ogdata[j, :]
                    data_sep_1 = np.vstack([data_sep_1, new_row])
                    break

        data_sep_1 = np.delete(data_sep_1, list_j, axis=1)
        #data_sep_1 = np.array([[np.double(x) for x in row] for row in data_sep_1])
        print('datasep1 prepared', data_sep_1)

        # Match fact_vary_rest rows with original factors
        for i in range(len(fact_vary_rest)):
            ck1 = fact_vary_rest[i]
            for j in range(len(factors)):
                ck2 = factors[j]
                if np.allclose(ck1, ck2, rtol=1e-8, atol=1e-8):
                    new_row = ogdata[j, :]
                    data_sep_2 = np.vstack([data_sep_2, new_row])
                    break

        data_sep_2 = np.delete(data_sep_2, list_i, axis=1)
        #data_sep_2 = np.array([[np.double(x) for x in row] for row in data_sep_2])
        print('datasep2 prepared', data_sep_2)

        # Save results
        try:
            os.makedirs("results/separable_mult/", exist_ok=True)
        except:
            pass

        str1 = filename + "-mult_a"
        str2 = filename + "-mult_b"

        np.savetxt("results/separable_mult/" + str1, np.round(data_sep_1, 5), fmt="%.5f", delimiter=" ", newline="\n", encoding="utf-8")
        np.savetxt("results/separable_mult/" + str2, np.round(data_sep_2, 5), fmt="%.5f", delimiter=" ", newline="\n", encoding="utf-8")

        return ("results/separable_mult/", str1, "results/separable_mult/", str2)

    except Exception as e:
        print("Error:", e)
        return (-1, -1)

'''
######################################################################################################################################
# I will remove the following part and search for fact_vary_one and fact_vary_rest in the original dataset to make the sub-dataset
######################################################################################################################################
                        
        # Abang -- the following part save data_sep_1 and data_sep_2 into two different files
        with torch.no_grad():
            str1 = filename+"-mult_a"
            str2 = filename+"-mult_b"
            # save the first half
            data_sep_1 = variables
            data_sep_1 = np.delete(data_sep_1,list_j,axis=1)
#Abang -- I need to change the last column too            
            data_sep_1 = np.column_stack((data_sep_1,model(fact_vary_one).cpu()))
            # save the second half  
            data_sep_2 = variables
            data_sep_2 = np.delete(data_sep_2,list_i,axis=1)
            data_sep_2 = np.column_stack((data_sep_2,model(fact_vary_rest).cpu()/model(fact_vary).cpu()))
            try:
                os.mkdir("results/separable_mult/")
            except:
                pass
            np.savetxt("results/separable_mult/"+str1,data_sep_1)
            np.savetxt("results/separable_mult/"+str2,data_sep_2)
            # if it is separable, return the 2 new files created and the index of the column with the separable variable
            return ("results/separable_mult/",str1,"results/separable_mult/",str2)

    except Exception as e:
        print(e)
        return (-1,-1)
'''

# update on 01/11/24 17:49
# 02/26/24 23:30 added constant output line 398

# update on 03/19/24 23:37 Replaced NN part by data-driven algorithm
