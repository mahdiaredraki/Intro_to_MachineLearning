# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:44:24 2020

@author: Mahdi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats

################################################################
#                     Question 2 - Part 1                      #
################################################################ 

### Loop for N_Exp number of experiments and save optimal number of GMM from BIC and K_fold CV
N_Exp=100

Best_M_BIC=np.zeros(N_Exp)
Best_M_Kfold=np.zeros(N_Exp)

for Loop in range(0,N_Exp):

    #------------ Setup the Given vectors and matrices ----------#
    
    ## Number of samples and dimensions ##
    n= 2; #Number of dimensions
    N= 1000; #Number of samples simulated
    C= 10; #Number of classes
    ##----------------------------------##
    
    ## GMM means and covariance ##
    Sigma=np.zeros((C,n,n))
    Mu=np.zeros((C,n))
    Weight=np.zeros(C)
    Random_Sigma=[0.5, 0.3, 0.7,0.4, 0.4, 0.8, 0.4,0.2,0.5,0.6]
    Mean=-C
    for i in range(0,C):
        Mean
        Sigma[i,:,:]=np.array([[Random_Sigma[i],                0],
                               [0,                Random_Sigma[i]]])
        Mu[i,:]=np.array([Mean,  0])
        Weight[i]=1/C
        Mean=Mean+2
    
    ##----------------------------##
    
    #--- Generate Validation Dataset based on class conditional pdfs ## 
    Fig_Num=1;
    def generateDataFromGMM(N,alpha,Mu,Sigma):
        # Generates N vector samples from the specified mixture of Gaussians
        # Returns samples and their component labels
        # Data dimensionality is determined by the size of mu/Sigma parameters
        
        n = len(Mu[0,:]); # Data dimensionality
        C = len(alpha); # Number of components
        x = np.zeros((n,N)); 
    
        # Decide randomly which samples will come from each component
        u = np.random.random((1,N)); 
        thresholds = np.cumsum(alpha);
        
        fig = plt.figure(Fig_Num)
        ax = plt.axes(projection='3d')
        
        FullRange=np.zeros((100,2))
        FullRange[:,0]=np.linspace(np.min(Mu[:,0])-3,np.max(Mu[:,0])+2,100)
        FullRange[:,1]=np.linspace(np.min(Mu[:,1])-3,np.max(Mu[:,1])+2,100)

        X_plot, Y_plot = np.meshgrid(FullRange[:,0],FullRange[:,1])
        XX = np.array([X_plot.ravel(), Y_plot.ravel()])
        for i in range(0,C):
            # indl = (t for t in enumerate(int(u[0,:])) if t<= thresholds[i]);
            indl = np.argwhere(u[0,:] <= thresholds[i])
            Nl = len(indl);
            u[0,indl[:,0]] = 1.1*np.ones((Nl)); # these samples should not be used again
            x[:,indl[:,0]] = np.transpose(np.random.multivariate_normal(Mu[i,:],Sigma[i,:,:],Nl));
            dist = stats.multivariate_normal(Mu[i,:],Sigma[i,:,:])
            PDF=dist.pdf(np.transpose(XX))
            PDF = PDF.reshape(X_plot.shape)
            
            ax.contour(X_plot,Y_plot,PDF,200)
            ax.set_xlabel("X1 Axis")
            ax.set_ylabel("X2 Axis")
            ax.set_zlabel("GMM Likelihood")
            print("done1")
        
        return x          
    #--- Generate Training Dataset based on class conditional pdfs ## 
    N_random_numbers=np.random.random(N);
    X = np.zeros((n, N)) # X Contains training dataset
    X= np.transpose(generateDataFromGMM(N,Weight,Mu[:,:],Sigma[:,:,:]))
    Data_Set=np.linspace(0,N,N)
    
    # Setup figure features
    Fig_Num=Fig_Num+1;
    X_Up_Lim=15;
    X_Low_Lim=-15;
    Y_Up_Lim=3;
    Y_Low_Lim=-3;
    
    #Plot training dataset for visualization
    fig=plt.figure(Fig_Num)
    m=['o']
    plt.scatter(X[:,0], X[:,1], c='#8c564b', marker=m[0], s=30, alpha=0.6)
    plt.title(str(N)+" samples generated")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim((X_Low_Lim, X_Up_Lim))
    plt.ylim((Y_Low_Lim, Y_Up_Lim))
    #---------- END Setup the Given vectors and matrices --------#
    
    #---BIC For GMM---#
    N_Params=20
    neg2logLikelihood=np.zeros(N_Params)
    BIC=np.zeros(N_Params)
    for M in range(1,N_Params+1):
        gmm=GaussianMixture(M,covariance_type='full')
        Best_gmm=gmm.fit(X[:,:])
        Est_Mu=Best_gmm.means_
        Est_Sigma=Best_gmm.covariances_
        Est_Weight=Best_gmm.weights_
        LogLikelihood=Best_gmm.score_samples(X)
        neg2logLikelihood[M-1] = -2*np.sum(LogLikelihood);
        BIC[M-1] = neg2logLikelihood[M-1] + M*np.log(N*n);
  
    Best_M_BIC[Loop]=np.argmin(BIC)+1
    
    #---Plot the BIC value for 1 Experiment with M from 1 to 20
    Range_M=np.linspace(1,M,M)
    Fig_Num=Fig_Num+1
    plt.figure(Fig_Num)
    plt.plot(Range_M,BIC)
    plt.scatter(Best_M_BIC,BIC[Best_M_BIC[Loop]-1])
    TEXT= "Optimal # of gaussians="+str(Best_M_BIC[Loop])
    plt.annotate(TEXT, xy=(Best_M_BIC[Loop], BIC[Best_M_BIC[Loop]-1]), xytext=(10, BIC[1]),arrowprops=dict(facecolor='black', shrink=0.1))
    plt.xlim((1,M))
    plt.xlabel("Number of Gaussian Distributions")
    plt.xticks(np.arange(1, M+1, step=1))
    plt.ylabel("BIC")
    plt.title("BIC values obtained using "+str(N)+" samples")
    #--END BIC For GMM---#  
    
    #---K-fold For GMM---#
    K_fold=5
    
    ## Divide dataset into K_fold sub-datasets
    N_perDataset=int(N/K_fold)
    Begin_Index=0
    End_Index=N_perDataset
    X_partitioned=np.zeros((K_fold,N_perDataset,n))
    for ITER in range(0,K_fold):
        X_partitioned[ITER,:,:]=X[Begin_Index:End_Index,:]
        Begin_Index=End_Index
        End_Index=End_Index+N_perDataset
    
    Sum_LogLikelihood=np.zeros(N_Params)
    for M in range(1,N_Params+1):
        for K in range(0,K_fold):
            X_Validate=np.zeros((N_perDataset,n))
            X_Train=np.zeros((N_perDataset*(K_fold-1),n))
            X_Validate[:,:]=X_partitioned[K,:,:]
            K_Train=(np.linspace(0,K_fold-1,K_fold))
            K_Train=np.delete(K_Train,K)
            Begin_Index=0
            End_Index=N_perDataset
            for K_temp in K_Train:
                K_temp=int(K_temp)
                X_Train[Begin_Index:End_Index,:]=X_partitioned[K_temp,:,:]
                Begin_Index=End_Index
                End_Index=End_Index+N_perDataset
            
            #EM for GMM
            gmm=GaussianMixture(M,covariance_type='full')
            Best_gmm=gmm.fit(X_Train)
            Est_Mu=Best_gmm.means_
            Est_Sigma=Best_gmm.covariances_
            Est_Weight=Best_gmm.weights_
            LogLikelihood=Best_gmm.score_samples(X_Validate)
            Sum_LogLikelihood[M-1]= Sum_LogLikelihood[M-1]+np.sum(LogLikelihood)
            
    Best_M_Kfold[Loop]=np.argmax(Sum_LogLikelihood)+1
    #---Plot the Kfold value for 1 Experiment with M from 1 to 20
    Fig_Num=Fig_Num+1
    plt.figure(Fig_Num)
    plt.plot(Range_M,Sum_LogLikelihood)
    plt.scatter(Best_M_Kfold[Loop],Sum_LogLikelihood[Best_M_Kfold[Loop]-1])
    TEXT= "Optimal # of gaussians="+str(Best_M_Kfold[Loop])
    plt.annotate(TEXT, xy=(Best_M_Kfold[Loop], Sum_LogLikelihood[Best_M_Kfold[Loop]-1]), xytext=(10, Sum_LogLikelihood[1]),arrowprops=dict(facecolor='black', shrink=0.1))
    plt.xlim((1,M))
    plt.xlabel("Number of Gaussian distributions")
    plt.xticks(np.arange(1, M+1, step=1))
    plt.ylabel("mean Log-Likelihood")
    plt.title("K_fold values obtained using "+str(N)+" samples") 
    #--END K-fold For GMM---#

##---- Plot histograms for estimated optimal number of Gaussian distributions using BIC and K-folg CV
Fig_Num=Fig_Num+1
plt.figure(Fig_Num)
plt.hist(Best_M_BIC,bins=np.arange(N_Params+1) - 0.5)
plt.xticks(range(N_Params))
plt.title("Histogram of number of GMM using BIC for: "+str(N)+" samples")

Fig_Num=Fig_Num+1
plt.figure(Fig_Num)
plt.hist(Best_M_Kfold,bins=np.arange(N_Params+1) - 0.5)
plt.xticks(range(N_Params))
plt.title("Histogram of number of GMs using K_fold CV for: "+str(N)+" samples")

