# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:44:24 2020

@author: Mahdi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:24:23 2020

@author: Mahdi
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm
from scipy.optimize import minimize

################################################################
#                          Question 1                          #
################################################################ 

#--- Set flag to run Logististic-LINEAR or Logisitic-QUADRATIC
Flag=0; # 0 for Linear, 1 for Quadratic

#--- Loop through training datasets
N_Loop=[1000]
Figure_COUNT=1
for N in N_Loop:
    Figure_COUNT=Figure_COUNT+20
    #------------ Setup the Given vectors and matrices ----------#
    
    ## Number of samples and dimensions ##
    n= 2; #Number of dimensions
    # N= 500; #Number of samples simulated
    N_Validate=20000;
    C= 2; #Number of classes
    ##----------------------------------##
    
    ## Class means and covariance ##
    Sigma_0=np.zeros((2,n,n));
    Sigma_0[0,:,:]=np.array([[4, 0],
                             [0, 4]])
    Sigma_0[1,:,:]=np.array([[1, 0],
                             [0, 3]])
    Mu_0=np.zeros((2,n))
    Mu_0[0,:]=np.array([5,  0])
    Mu_0[1,:]=np.array([0,  4])
    Weight=[0.5,0.5]
    
    Sigma_1=np.zeros((n,n));
    Sigma_1=np.array([[2, 0],
                      [0, 2]])
    Mu_1=np.zeros((1,n))
    Mu_1[0,:]=np.array([3,  2])
#    Mu_1[0,:]=np.array([-10,  -5])
    ##----------------------------##
    
    ## Class priors ##
    p=np.array([0.6, 0.4]); #p[0]=class 0 prior, p[1]=class 1 prior
    ##--------------##
    
    
    #--- Generate Validate Dataset based on class conditional pdfs ## 
    N_random_numbers_Validate=np.random.random(N_Validate);
    label_Validate=np.zeros(N_Validate);
    for i in range(0,N_Validate):
        if N_random_numbers_Validate[i]<=p[0]:
            label_Validate[i]=0;
        else:
            label_Validate[i]=1;
    Sum_CL_Validate=np.zeros(C)  
    for i in range(0,C):      
        Sum_CL_Validate[i]=sum(label_Validate==i); #Number of data points with the given class label
    
    X_Validate = np.zeros((n, N_Validate))
    L_Counter_Validate=np.zeros(C);
    
    Data_Set_L_Validate=np.zeros((C,N_Validate,n))
    
    def generateDataFromGMM(N,alpha,Mu,Sigma):
        # Generates N vector samples from the specified mixture of Gaussians
        # Returns samples and their component labels
        # Data dimensionality is determined by the size of mu/Sigma parameters
        
        n = len(Mu[:,0]); # Data dimensionality
        C = len(alpha); # Number of components
        x = np.zeros((n,N)); 
        labels = np.zeros((1,N)); 
        # Decide randomly which samples will come from each component
        u = np.random.random((1,N)); 
        thresholds = np.cumsum(alpha);
        for i in range(0,C):
            # indl = (t for t in enumerate(int(u[0,:])) if t<= thresholds[i]);
            indl = np.argwhere(u[0,:] <= thresholds[i])
            Nl = len(indl);
            labels[0,indl[:,0]] = i*np.ones((Nl));
            u[0,indl[:,0]] = 1.1*np.ones((Nl)); # these samples should not be used again
            x[:,indl[:,0]] = np.transpose(np.random.multivariate_normal(Mu[i,:],Sigma[i,:,:],Nl));
        
        return x
    
    Data_Set_L_Validate[0,0:int(Sum_CL_Validate[0]),:]= np.transpose(generateDataFromGMM(int(Sum_CL_Validate[0]),Weight,Mu_0[:,:],Sigma_0[:,:,:]))
    Data_Set_L_Validate[1,0:int(Sum_CL_Validate[1]),:]= (np.random.multivariate_normal(Mu_1[0,:],Sigma_1[:,:],int(Sum_CL_Validate[1])))
    for i in range(0,N_Validate):
        for CLASS in range(0,C):
            if label_Validate[i]==CLASS:
                X_Validate[:,i]= np.transpose(Data_Set_L_Validate[CLASS,int(L_Counter_Validate[CLASS]),:]);
                L_Counter_Validate[CLASS]=L_Counter_Validate[CLASS]+1;
                
    #--- Generate Training Dataset based on class conditional pdfs ## 
    N_random_numbers=np.random.random(N);
    label=np.zeros(N);
    for i in range(0,N):
        if N_random_numbers[i]<=p[0]:
            label[i]=0;
        else:
            label[i]=1;
    Sum_CL=np.zeros(C)  
    for i in range(0,C):      
        Sum_CL[i]=sum(label==i); #Number of data points with the given class label
    
    X = np.zeros((n, N))
    L_Counter=np.zeros(C);
    
    Data_Set_L=np.zeros((C,N,n))
    Data_Set_L[0,0:int(Sum_CL[0]),:]= np.transpose(generateDataFromGMM(int(Sum_CL[0]),Weight,Mu_0[:,:],Sigma_0[:,:,:]))
    Data_Set_L[1,0:int(Sum_CL[1]),:]= (np.random.multivariate_normal(Mu_1[0,:],Sigma_1[:,:],int(Sum_CL[1])))
    for i in range(0,N):
        for CLASS in range(0,C):
            if label[i]==CLASS:
                X[:,i]= np.transpose(Data_Set_L[CLASS,int(L_Counter[CLASS]),:]);
                L_Counter[CLASS]=L_Counter[CLASS]+1;
            
    Fig_Num=Figure_COUNT;
    Up_Lim=12;
    Low_Lim=-6;
    
    fig=plt.figure(Fig_Num)
    m=['o','s']
    plt.scatter(Data_Set_L[0,:,0], Data_Set_L[0,:,1], c='#8c564b', marker=m[0], s=30, alpha=0.6)
    plt.scatter(Data_Set_L[1,:,0], Data_Set_L[1,:,1], c='#2ca02c', marker=m[1], s=30, alpha=0.6)
    plt.legend(['Class 0', 'Class 1'])
    plt.title(str(N)+" samples generated")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim((Low_Lim, Up_Lim))
    plt.ylim((Low_Lim, Up_Lim))
    
    #---------- END Setup the Given vectors and matrices --------#
    
    #--- Setup optimization function and parameters ----#
    def ModelFunction_Linear(X,W):
        [n,N]=np.shape(X)
        Z=np.zeros((n+1,N))
        Z[0,:]=np.ones(N)
        Z[1,:]=X[0,:]
        Z[2,:]=X[1,:]
        VALUE=np.ones(N)/((np.ones(N)+np.exp(-np.matmul(np.transpose(W),Z))))
        return VALUE
    
    def ModelFunction_Quad(X,W):
        [n,N]=np.shape(X)
        Z=np.zeros((6,N))
        Z[0,:]=np.ones(N)
        Z[1,:]=X[0,:]
        Z[2,:]=X[1,:]
        Z[3,:]=X[0,:]*X[0,:]
        Z[4,:]=X[0,:]*X[1,:]
        Z[5,:]=X[1,:]*X[1,:]
        
        VALUE=np.ones(N)/((np.ones(N)+np.exp(-np.matmul(np.transpose(W),Z))))
        return VALUE
    
    def CostFunction(W,X,label):
        [n,N]=np.shape(X)
        if Flag==0:
            SUM=np.sum( label*np.log(ModelFunction_Linear(X,W)) + (1-label)*np.log(1-ModelFunction_Linear(X,W))  )
            Cost= -(1/N)*SUM
        elif Flag==1:
            SUM=np.sum( label*np.log(ModelFunction_Quad(X,W)) + (1-label)*np.log(1-ModelFunction_Quad(X,W))  )
            Cost= -(1/N)*SUM
#        print("Cost: " +str(Cost))
        return Cost
    if Flag==0:
        Guess=[0, 0 , 0]
    elif Flag==1:
        Guess=[0, 0 , 0, 0, 0, 0]
        
    result = minimize(CostFunction,Guess,args=(X,label),options={'maxiter':200},tol= 1e-06) 
    
    if Flag==0:
        Lin_Model_params=result.x
        print("Guess for linear fit parameters: w_0="+str(Lin_Model_params[0])+" w_1="+str(Lin_Model_params[1])+" w_2="+str(Lin_Model_params[2]))
        Lin_Fit_Range=np.linspace(Low_Lim,Up_Lim,N) #Picking values for x[0,:]
        Log_Lin_Model=(-1/Lin_Model_params[2])*(Lin_Model_params[0]*np.ones(N)+Lin_Model_params[1]*Lin_Fit_Range) #Calculating values of x[1,:]
        plt.figure(Fig_Num)
#        plt.plot(Lin_Fit_Range,Log_Lin_Model)
    elif Flag==1:
        Quad_Model_params=result.x
        print("Guess for Quadratic fit parameters: w_0="+str(Quad_Model_params[0])+" w_1="+str(Quad_Model_params[1])+" w_2="+str(Quad_Model_params[2])+" w_3="+str(Quad_Model_params[3])+" w_4="+str(Quad_Model_params[4])+" w_5="+str(Quad_Model_params[5]))
        
    #---- Calculate class pdf ----#
    if Flag==0:
        Est_pxgivenL=np.zeros((C,N_Validate))
        Est_pxgivenL[0,:] =1-ModelFunction_Linear(X_Validate,Lin_Model_params)             #Evaluate p(x|L=0)
        Est_pxgivenL[1,:] =  ModelFunction_Linear(X_Validate,Lin_Model_params)              #Evaluate p(x|L=1)
    elif Flag==1:
        Est_pxgivenL=np.zeros((C,N_Validate))
        Est_pxgivenL[0,:] =1-ModelFunction_Quad(X_Validate,Quad_Model_params)             #Evaluate p(x|L=0)
        Est_pxgivenL[1,:] =  ModelFunction_Quad(X_Validate,Quad_Model_params)              #Evaluate p(x|L=1)
    #- END Calculate class pdf --#
    
    #---- Calculate Discriminant Score ----#
    Est_discriminantScore=np.zeros(N_Validate);
    
    for i in range(0,N_Validate):
         Est_discriminantScore[i] = math.log(Est_pxgivenL[1,i])-math.log(Est_pxgivenL[0,i]);
         # If discriminantScore > gamma, D=1, otherwise, D=0
    #- END Calculate Discriminant Score --#
    Est_gamma = np.sort(Est_discriminantScore);
    eps=0.1;
    Est_mid_gamma=np.zeros(N_Validate);
    Est_mid_gamma[0]=Est_gamma[0]-eps;
    Est_mid_gamma[N_Validate-1]=Est_gamma[N_Validate-1]+eps;
    for i in range(1,N_Validate-1):
        Est_mid_gamma[i]=(Est_gamma[i-1]+Est_gamma[i])/2;
    
    SZ= np.size(Est_mid_gamma);
    Est_pTN=np.zeros(SZ);
    Est_pFN=np.zeros(SZ);
    Est_pTP=np.zeros(SZ);
    Est_pFP=np.zeros(SZ);
    
    for i in range(0,SZ):
        Est_Decision=np.zeros(N_Validate);
        TP = np.zeros(1); #Number of Correct Decisions/ True Positive
        FP =np.zeros(1); #Number of False Alarms/False Positive
        TN=np.zeros(1); #Number of True Negatives
        FN=np.zeros(1); #Number of False Negatives
        for j in range(0,N_Validate):
            Est_Decision[j]=(Est_discriminantScore[j]>=Est_mid_gamma[i]);
            if Est_Decision[j]==1 and label_Validate[j]==1:
                TP = TP+1;
            elif  Est_Decision[j]==1 and label_Validate[j]==0:
                 FP=FP+1;
            elif  Est_Decision[j]==0 and label_Validate[j]==0:
                 TN=TN+1;
            else:  
                 FN=FN+1;
    
        Est_pFP[i]=FP/Sum_CL_Validate[0];
        Est_pTP[i]=TP/Sum_CL_Validate[1];
        Est_pFN[i]=FN/Sum_CL_Validate[1];
        Est_pTN[i]=TN/Sum_CL_Validate[0];
    
    ## Calculate area under ROC curve
    Flip_Est_pTP=np.flip(Est_pTP);
    Flip_Est_pFP=np.flip(Est_pFP);
    Area_Q1_P2=0;
    for i in range(1,N_Validate):
        Area_Q1_P2= Area_Q1_P2 + Flip_Est_pTP[i]*(Flip_Est_pFP[i]-Flip_Est_pFP[i-1]);
    
    #Calculate  the  probability of error as a function of gamma
    Est_P_Err=np.zeros(SZ);
    for i in range(0,SZ):
        Est_P_Err[i]=Est_pFP[i]*p[0]+Est_pFN[i]*p[1];
    Est_Min_P_Err=min(Est_P_Err);
    Est_Min_P_Err_ind = [i for i, v in enumerate(Est_P_Err) if v == Est_Min_P_Err];
    Est_Best_Gamma=Est_mid_gamma[Est_Min_P_Err_ind];
    
    Fig_Num=Fig_Num+1;
    plt.figure(Fig_Num)
    plt.plot(Est_pFP,Est_pTP)
    plt.xlabel("pFA")
    plt.ylabel("pTP")
    if Flag==0:
        plt.title("ROC curve for Q1 part 3 - Log-Linear func - "+str(N)+" training samples")
    elif Flag==1:
        plt.title("ROC curve for Q1 part 3 - Log-Quadratic func - "+str(N)+" training samples")
        
    plt.scatter(Est_pFP[Est_Min_P_Err_ind],Est_pTP[Est_Min_P_Err_ind],50,'#d62728')
    TEXT= "Min pError="+str(np.round(Est_Min_P_Err,4))+"\n Threshold value= "+str(np.round(Est_Best_Gamma,4))+"\n pTP= "+str(np.round(Est_pTP[Est_Min_P_Err_ind],4))+"\n pFP= "+str(np.round(Est_pFP[Est_Min_P_Err_ind],4))+"\n Area under curve= "+str(np.round(Area_Q1_P2,4))
    plt.annotate(TEXT, xy=(Est_pFP[Est_Min_P_Err_ind], Est_pTP[Est_Min_P_Err_ind]), xytext=(0.35, 0.4),arrowprops=dict(facecolor='black', shrink=0.1))
    plt.show()
    
    Fig_Num=Fig_Num+1;
    plt.figure(Fig_Num)
    plt.plot(Est_mid_gamma,Est_P_Err)
    plt.xlabel("Threshold")
    plt.ylabel("Probability of Error")
    if Flag==0:
        plt.title("p(error) vs Threshold - Q1 part 3 - Log-Linear func - "+str(N)+" training samples")
    elif Flag==1:
        plt.title("p(error) vs Threshold - Q1 part 3 - Log-Quadratic func - "+str(N)+" training samples")
        
    Est_Color=[]
    Est_COLOR_L0=[]
    Est_COLOR_L1=[]
    Est_Indeces_L0=np.zeros(int(Sum_CL_Validate[0]),dtype=int); #Contains the index numbers of Dataset X that has label 1
    Est_Indeces_L1=np.zeros(int(Sum_CL_Validate[1]),dtype=int);
    Est_L_Counter=np.zeros(C,dtype=int);
    
    for i in range(0,N_Validate):
        Est_Decision[i]=Est_discriminantScore[i]>=Est_Best_Gamma
        if Est_Decision[i]==label_Validate[i]:
            Est_Color.append('green');
        else:
            Est_Color.append('red');
        if label_Validate[i]==0:
            Est_COLOR_L0.append(Est_Color[i]);
            Est_Indeces_L0[Est_L_Counter[0]]=i;
            Est_L_Counter[0]=Est_L_Counter[0]+1;
        elif label_Validate[i]==1:
            Est_COLOR_L1.append(Est_Color[i]);
            Est_Indeces_L1[Est_L_Counter[1]]=i;
            Est_L_Counter[1]=Est_L_Counter[1]+1;
    
                
    Fig_Num=Fig_Num+1;
    plt.figure(Fig_Num)     
    
    plt.scatter(X_Validate[0,Est_Indeces_L0],X_Validate[1,Est_Indeces_L0],facecolors='none', edgecolors=Est_COLOR_L0,marker=m[0], s=25, alpha=0.7)
    plt.scatter(X_Validate[0,Est_Indeces_L1],X_Validate[1,Est_Indeces_L1],facecolors='none', edgecolors=Est_COLOR_L1,marker=m[1], s=25, alpha=0.7)
    if Flag==0:
#        plt.plot(Lin_Fit_Range,Log_Lin_Model)
        plt.title("Q1 part 3 - Log-Linear func - Classification of both classes Using "+str(N)+" training samples")
    elif Flag==1:
        plt.title("Q1 part 3 - Log-Quadratic func - Classification of both classes Using "+str(N)+" training samples")
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X2',fontsize=12)
    plt.xlim((Low_Lim, Up_Lim))
    plt.ylim((Low_Lim, Up_Lim))
    plt.axis('square')
        
    
    Fig_Num=Fig_Num+1;
    plt.figure(Fig_Num)
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    for i in range(0,C):
        if i==0:
            SUBPLOT=1;
            Est_Scat_Color=Est_COLOR_L0
            Scat_Indeces=Est_Indeces_L0
            Marker=m[0]
        else:
            SUBPLOT=2;
            Est_Scat_Color=Est_COLOR_L1
            Scat_Indeces=Est_Indeces_L1
            Marker=m[1]
            
        plt.subplot(1,2,SUBPLOT)
        if Flag==0:
#            plt.plot(Lin_Fit_Range,Log_Lin_Model)
            plt.title("Q1 part 3 - Log-Linear func - CL= "+str(int(i)))
        elif Flag==1:
            plt.title("Q1 part 3 - Log-Quadratic func - CL= "+str(int(i)))
        plt.scatter(X_Validate[0,Scat_Indeces],X_Validate[1,Scat_Indeces],facecolors='none', edgecolors=Est_Scat_Color,marker=Marker, s=15, alpha=0.7)
        plt.xlabel('X1',fontsize=12)
        plt.ylabel('X2',fontsize=12)
        plt.axis('square')
        plt.xlim((Low_Lim, Up_Lim))
        plt.ylim((Low_Lim, Up_Lim))
        plt.grid(b=True,which='both',axis='both')
        
        plt.subplot(1,2,SUBPLOT)
        if Flag==0:
#            plt.plot(Lin_Fit_Range,Log_Lin_Model)
            plt.title("Q1 part 3 - Log-Linear func - CL= "+str(int(i)))
        elif Flag==1:
            plt.title("Q1 part 3 - Log-Quadratic func - CL= "+str(int(i)))
        plt.scatter(X_Validate[0,Scat_Indeces],X_Validate[1,Scat_Indeces],facecolors='none', edgecolors=Est_Scat_Color,marker=Marker, s=15, alpha=0.7)
        plt.xlabel('X1',fontsize=12)
        plt.ylabel('X2',fontsize=12)
        plt.axis('square')
        plt.xlim((Low_Lim, Up_Lim))
        plt.ylim((Low_Lim, Up_Lim))
        plt.grid(b=True,which='both',axis='both')
    
    print("Finished Q1 part 3")

    
    
                         
                         
    
    
 