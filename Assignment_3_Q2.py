# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:27:48 2020

@author: mahdiar
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#------ Generate Data------#
n=7  #Number of dimensions
N_Train=100 # Training Datset sample sizes
N_Test=10000 # Testing Dataset sample size

## Model Parameters ##
Arbit_vals=(np.random.random(n*n+7))/2
Sigma=np.zeros((n,n))
ROW=0
for i in range(0,n*n,7):
    Sigma[ROW,0:7]=Arbit_vals[i:i+7]
    ROW =ROW+1
Mu=np.array([Arbit_vals[49],  Arbit_vals[50], Arbit_vals[51], Arbit_vals[52], Arbit_vals[53], Arbit_vals[54], Arbit_vals[55]]);
N_Alpha=500
alpha=np.logspace(-4,3,N_Alpha)
Best_Beta=np.zeros(N_Alpha)
Neg2LogLikelihood=np.zeros(N_Alpha)

for Alpha_Index in range(0,N_Alpha):
    a= np.random.random(n)
    
    v_Train=np.random.normal(0,1,N_Train)
    z_Train=np.transpose(np.random.multivariate_normal(np.zeros(n),np.identity(n)*alpha[Alpha_Index],N_Train))
    v_Test=np.random.normal(0,1,N_Test)
    z_Test=np.transpose(np.random.multivariate_normal(np.zeros(n),np.identity(n)*alpha[Alpha_Index],N_Test))
    
    ## Generate training samples ## 
    x_Train= np.transpose(np.random.multivariate_normal(Mu,Sigma,N_Train))
    x_Test= np.transpose(np.random.multivariate_normal(Mu,Sigma,N_Test))
    X_Train=np.vstack((np.ones(N_Train),x_Train))
    X_Test=np.vstack((np.ones(N_Test),x_Test))
    ## Calculate y based on generated samples ##
    y_Train=np.transpose(a)@(x_Train+z_Train)+v_Train
    y_Test=np.transpose(a)@(x_Test+z_Test)+v_Test
    #------ END Generate Data----#
    
    
    #---K-fold ---#
    K_fold=10
    
    ## Divide dataset into K_fold sub-datasets
    N_perDataset=int(N_Train/K_fold)
    Begin_Index=0
    End_Index=N_perDataset
    X_partitioned=np.zeros((K_fold,n+1,N_perDataset))
    y_partitioned=np.zeros((K_fold,N_perDataset))
    for ITER in range(0,K_fold):
        X_partitioned[ITER,:,:]=X_Train[:,Begin_Index:End_Index]
        y_partitioned[ITER,:]=y_Train[Begin_Index:End_Index]
        Begin_Index=End_Index
        End_Index=End_Index+N_perDataset
    
    N_Params=500 #Number of Beta values to test
    Beta=np.logspace(-4,2,N_Params)
    Sum_LogLikelihood=np.zeros(N_Params)
    for M in range(0,N_Params):
        for K in range(0,K_fold):
            X_Test_kfold=np.zeros((n+1,N_perDataset))
            X_Train_kfold=np.zeros((n+1,N_perDataset*(K_fold-1)))
            y_Test_kfold=np.zeros((N_perDataset))
            y_Train_kfold=np.zeros((N_perDataset*(K_fold-1)))
            X_Test_kfold[:,:]=X_partitioned[K,:,:]
            y_Test_kfold[:]=y_partitioned[K,:]
            K_Train=(np.linspace(0,K_fold-1,K_fold))
            K_Train=np.delete(K_Train,K)
            Begin_Index=0
            End_Index=N_perDataset
            for K_temp in K_Train:
                K_temp=int(K_temp)
                X_Train_kfold[:,Begin_Index:End_Index]=X_partitioned[K_temp,:,:]
                y_Train_kfold[Begin_Index:End_Index]=y_partitioned[K_temp,:]
                Begin_Index=End_Index
                End_Index=End_Index+N_perDataset
            
            #max-log-likelihood calculation
            w_hat=1/(1/Beta[M]+np.sum(X_Train_kfold@np.transpose(X_Train_kfold),axis=0))*np.sum(y_Train_kfold*X_Train_kfold,axis=1)
            LogLikelihood=-1/2*(np.sum(y_Train_kfold)-np.sum(np.transpose(w_hat)@X_Train_kfold) )**2
            Sum_LogLikelihood[M-1]= Sum_LogLikelihood[M-1]+np.sum(LogLikelihood)
            
    Mean_Loglikelihood=Sum_LogLikelihood/K_fold
    Best_M_Kfold=np.argmax(Mean_Loglikelihood)
    Best_Beta[Alpha_Index]=Beta[Best_M_Kfold]
    Fig_Num=1
    plt.figure(Fig_Num)
    plt.plot(Beta,Mean_Loglikelihood)
    plt.scatter(Best_Beta[Alpha_Index],Mean_Loglikelihood[int(Best_M_Kfold)])
    TEXT= "Optimal value for Beta="+str(round(Best_Beta[Alpha_Index],4))
    plt.annotate(TEXT, xy=(Best_Beta[Alpha_Index], Mean_Loglikelihood[int(Best_M_Kfold)]), xytext=(0.1, Mean_Loglikelihood[2]),arrowprops=dict(facecolor='black', shrink=0.1))
    # plt.xlim((1,M))
    plt.xlabel("Value of Beta")
    # plt.xticks(np.arange(1, M+1, step=1))
    plt.ylabel("Mean Loglikelihood")
    plt.title("K_fold values obtained using "+str(N_Train)+" samples") 
    plt.xscale("log")
    #---END K-fold ---#
    
    #------ Model Optimization -----#
    w_hat_Optimum=1/(1/Best_Beta[Alpha_Index]+np.sum(X_Train@np.transpose(X_Train),axis=0))*np.sum(y_Train*X_Train,axis=1)
    Neg2LogLikelihood[Alpha_Index]=(np.sum(y_Test)-np.sum(np.transpose(w_hat_Optimum)@X_Test))**2
    
Fig_Num=Fig_Num+1
plt.figure(Fig_Num)
plt.scatter(alpha,Neg2LogLikelihood)
plt.title("Plot of alpha vs. -2*Loglikelihood")
plt.ylabel("-2*Loglikelihood")
plt.xlabel("Alpha")
plt.xscale("log")
plt.yscale("log")

Fig_Num=Fig_Num+1
plt.figure(Fig_Num)
plt.scatter(alpha,Best_Beta)
plt.title("Plot of alpha vs. Optimum Beta")
plt.ylabel("Optimum Beta")
plt.xlabel("Alpha")
plt.xscale("log")
plt.yscale("log")



