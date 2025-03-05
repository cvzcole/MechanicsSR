from __future__ import print_function
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
            
def do_separability_plus(pathdir, filename, list_i,list_j):
    try:

        # load the data
        n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                variables = np.column_stack((variables,v))
            for j in range(1,n_variables+1):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                ogdata = np.column_stack((variables,v))  # original dataset in numpy form
        
        print('check here ogdata',variables)

        f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables) 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        #model.load_state_dict(torch.load(pathdir_weights+filename+".h5"))
        #model.eval()

        # make some variables at the time equal to the median of factors
        
        fact_vary = factors.clone()

        
        for k in range(len(factors[0])):
            fact_vary[:,k] = torch.full((len(factors),),torch.median(factors[:,k]))
        fact_vary_one = factors.clone()
        fact_vary_rest = factors.clone()
        for t1 in list_j:
            fact_vary_one[:,t1] = torch.full((len(factors),),torch.median(factors[:,t1]))
        for t2 in list_i:
            fact_vary_rest[:,t2] = torch.full((len(factors),),torch.median(factors[:,t2]))
        np_fact_one = fact_vary_one.numpy() # transfer to numpy form
        np_fact_rest = fact_vary_rest.numpy()   
        data_sep_1 = np.empty((0,n_variables+1))
        data_sep_2 = np.empty((0,n_variables+1))
        with torch.no_grad():
            str1 = filename+"-add_a"
            str2 = filename+"-add_b"
            # Save the first halfbool
            check_sub_1 = fact_vary_one

            for i in range(len(check_sub_1)):
                for j in range(len(factors)):
                    ck1=check_sub_1[i]
                    ck2=factors[j]
#                    print('okhere',torch.equal(ck1,ck2))
                    if (torch.equal(ck1,ck2)):
                        new_row= ogdata[j,:]
                        #print("original datapoint found",new_row)
                        data_sep_1 = np.vstack([new_row,data_sep_1])
                        #print('1added',ck1.unsqueeze(0))
                        break
            data_sep_1 = np.delete(data_sep_1, list_j, axis=1)
            print('addtive datasep1 prepared',data_sep_1)
            # Save the second half
            check_sub_2 = fact_vary_rest

            for i in range(len(check_sub_2)):
                for j in range(len(factors)):
                    ck1=check_sub_2[i]
                    ck2=factors[j]
                    #print('okhere',torch.equal(ck1,ck2))
                    if (torch.equal(ck1,ck2)):
                        new_row= ogdata[j,:]
                        #print("original datapoint found",new_row)
                        data_sep_2 = np.vstack([new_row,data_sep_2])
                        #print('1added',ck1.unsqueeze(0))
                        break
            data_sep_2 = np.delete(data_sep_2, list_i, axis=1)
            print('additive datasep2 prepared',data_sep_2)
            try:
                os.mkdir("results/separable_add/")
            except:
                pass
            np.savetxt("results/separable_add/"+str1,data_sep_1)
            np.savetxt("results/separable_add/"+str2,data_sep_2)
            # if it is separable, return the 2 new files created and the index of the column with the separable variable
            return ("results/separable_add/",str1,"results/separable_add/",str2)

    except Exception as e:
        print(e)
        return (-1,-1)
                 
                  
def do_separability_multiply(pathdir, filename, list_i,list_j):
    try:
        # load the data
        n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+filename, usecols=(0,))

        if n_variables==1:
            print(filename, "just one variable for ADD")
            # if there is just one variable you have nothing to separate
            return (-1,-1,-1)
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                variables = np.column_stack((variables,v))
            for j in range(1,n_variables+1):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                ogdata = np.column_stack((variables,v))  # original dataset in numpy form
        
        print('check here ogdata',variables)
        

        f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))  # original output in n*1 numpy array

        factors = torch.from_numpy(variables)  # factors are variables in tensor form

        factors = factors.float()

        product = torch.from_numpy(f_dependent)

        product = product.float()

        # load the trained model and put it in evaluation mode

        # make some variables at the time equal to the median of factors             
        # Abang -- the following code use NN to predict the values
        # ！！！ Abang -- I need to replace this part, and save the dataset into fact_vary_one and fact_vary_rest
        # factors are variables only, form changed here: transfer variables (numpy) to factors(torch)
        fact_vary = factors.clone()
        for k in range(len(factors[0])):
            fact_vary[:,k] = torch.full((len(factors),),torch.median(factors[:,k]))
        # Abang --fact_vary is a tensor filled with all mean values
        
        fact_vary_one = factors.clone()  # replace only list_j columns to mean value
        fact_vary_rest = factors.clone() # replace only list_i columns to mean value
        np_fact_one = fact_vary_one.numpy() # transfer to numpy form
        np_fact_rest = fact_vary_rest.numpy() 
        #print('done',np_fact_one)

        for t1 in list_j:
            fact_vary_one[:,t1] = torch.full((len(factors),),torch.median(factors[:,t1]))
        for t2 in list_i:
            fact_vary_rest[:,t2] = torch.full((len(factors),),torch.median(factors[:,t2]))  
        print('constants here',torch.median(factors[:,t1]),torch.median(factors[:,t2]))           
        with torch.no_grad():
            str1 = filename + "-mult_a"
            str2 = filename + "-mult_b"
        # Initialize empty arrays
        data_sep_1 = np.empty((0,n_variables+1))
        data_sep_2 = np.empty((0,n_variables+1))
        with torch.no_grad():
            str1 = filename + "-mult_a"
            str2 = filename + "-mult_b"

            # Save the first halfbool
            check_sub_1 = fact_vary_one

            for i in range(len(check_sub_1)):
                for j in range(len(factors)):
                    ck1=check_sub_1[i]
                    ck2=factors[j]
                    #print('okhere',torch.equal(ck1,ck2))
                    if (torch.equal(ck1,ck2)):
                        new_row= ogdata[j,:]
                        #print("original datapoint found",new_row)
                        data_sep_1 = np.vstack([new_row,data_sep_1])
                        #print('1added',ck1.unsqueeze(0))
                        break
            data_sep_1 = np.delete(data_sep_1, list_j, axis=1)
            print('datasep1 prepared',data_sep_1)
#            print('data_sep_1',data_sep_1)
            # Save the second half

            # Save the first halfbool
            check_sub_2 = fact_vary_rest

            for i in range(len(check_sub_2)):
                for j in range(len(factors)):
                    ck1=check_sub_2[i]
                    ck2=factors[j]
                    #print('okhere',torch.equal(ck1,ck2))
                    if (torch.equal(ck1,ck2)):
                        new_row= ogdata[j,:]
                        #print("original datapoint found",new_row)
                        data_sep_2 = np.vstack([new_row,data_sep_2])
                        #print('1added',ck1.unsqueeze(0))
                        break
            data_sep_2 = np.delete(data_sep_2, list_i, axis=1)
            print('datasep2 prepared',data_sep_2)
                    

#Abang -- I need to change the last column too
# insert searching algorithm or .where here!            
#            data_sep_1 = np.column_stack((data_sep_1,model(fact_vary_one).cpu()))
            # save the second half  
#            data_sep_2 = variables
#            data_sep_2 = np.delete(data_sep_2,list_i,axis=1)
#            data_sep_2 = np.column_stack((data_sep_2,model(fact_vary_rest).cpu()/model(fact_vary).cpu()))
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