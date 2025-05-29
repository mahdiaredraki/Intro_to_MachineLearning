# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:56:24 2020

@author: Mahdi
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
################################################################
#                          Question 1                          #
################################################################ 

#------------ Setup the Given vectors and matrices ----------#

## Number of samples and dimensions ##
n= 4; #Number of dimensions
N= 10000; #Number of samples simulated
##----------------------------------##

## Class means and covariance ##
Sigma_0=np.array([[2,       -0.5,   0.3,    0],
                  [-0.5,    1,      -0.5,   0],
                  [0.3,     -0.5,   1,      0],
                  [0,       0,      0,      2]]);

Sigma_1=np.array([[1,       0.3,    -0.2,   0],
                  [0.3,     2,      0.3,    0],
                  [-0.2,    0.3,    1,      0],
                  [0,       0,      0,      3]]);

Mu_0=np.array([-1, -1, -1, -1]);
Mu_1=np.array([ 1,  1,  1,  1]);
##----------------------------##

## Class priors ##
p=np.array([0.7, 0.3]); #First component for p(L=0), second component for p(L=1)
##--------------##

## Generate samples based on class conditional pdfs ## !!! ONLY RUN ONCE AND THEN COMMENT OUT
label=(np.random.random(N)>=p[1]).astype(int);
Sum_L1=sum(On for On in label if On ==1) ;
Sum_L0=N-Sum_L1 ;
X = np.zeros((n, N))
L0_Counter=0;
L1_Counter=0;
#
Data_Set_L0= np.random.multivariate_normal(Mu_0,Sigma_0,Sum_L0);
Data_Set_L1= np.random.multivariate_normal(Mu_1,Sigma_1,Sum_L1);
for i in range(0,N):
    if label[i]==0:
        X[:,i]= np.transpose(Data_Set_L0[L0_Counter,:]);
        L0_Counter=L0_Counter+1;
    else:
        X[:,i]= np.transpose(Data_Set_L1[L1_Counter,:]);
        L1_Counter=L1_Counter+1;
#np.save("HW1_1000Samples_X_values",X,allow_pickle=True)
#np.save("HW1_1000Samples_label_values",label,allow_pickle=True)

#X = np.load("HW1_1000Samples_X_values.npy")
#label = np.load("HW1_1000Samples_label_values.npy")
#Sum_L1=sum(On for On in label if On ==1) ;
#Sum_L0=N-Sum_L1 ;
Fig_Num=1;
plt.figure(Fig_Num)

plt.subplot(211)
plt.scatter(Data_Set_L0[:,0],Data_Set_L0[:,1],2,'red')
plt.scatter(Data_Set_L1[:,0],Data_Set_L1[:,1],2,'blue')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2-D Representation of the Training Data')

plt.subplot(212)
plt.scatter(Data_Set_L0[:,2],Data_Set_L0[:,3],2,'red')
plt.scatter(Data_Set_L1[:,2],Data_Set_L1[:,3],2,'blue')
plt.xlabel('Dimension 3')
plt.ylabel('Dimension 4')
print("Finished Q1 Section-A Part-1");
##-------------------------------------------------##
#---------- END Setup the  Given vectors and matrices --------#

##------- Q1 Part 2---------##
#---- Calculate Discriminant Score ----#
def evalGaussian(x, Mu, Sigma):
    g=multivariate_normal.pdf(x,Mu,Sigma);
    return g
discriminantScore=np.zeros(N);

for i in range(0,N):
     discriminantScore[i] = (evalGaussian(X[:,i],Mu_1[:],Sigma_1[:,:])/evalGaussian(X[:,i],Mu_0[:],Sigma_0[:,:]));
     # If discriminantScore > gamma, D=1, otherwise, D=0
#- END Calculate Discriminant Score --#
gamma = np.sort(discriminantScore);
eps=1;
mid_gamma=np.zeros(N);
mid_gamma[0]=gamma[0]-eps;
mid_gamma[N-1]=gamma[N-1]+eps;
for i in range(1,N-1):
    mid_gamma[i]=(gamma[i]+gamma[i+1])/2;

SZ= np.size(mid_gamma);
pTN=np.zeros(SZ);
pFN=np.zeros(SZ);
pTP=np.zeros(SZ);
pFP=np.zeros(SZ);
Area_2=np.zeros(1);


for i in range(0,SZ):
    Decision=np.zeros(N);
    TP = np.zeros(1); #Number of Correct Decisions/ True Positive
    FP =np.zeros(1); #Number of False Alarms/False Positive
    TN=np.zeros(1); #Number of True Negatives
    FN=np.zeros(1); #Number of False Negatives
    for j in range(0,N):
        Decision[j]=(discriminantScore[j]>=mid_gamma[i]);
        if Decision[j]==True and label[j]==1:
            TP = TP+1;
        elif  Decision[j]==True and label[j]==0:
             FP=FP+1;
        elif  Decision[j]==False and label[j]==0:
             TN=TN+1;
        else:  
             FN=FN+1;

    pFP[i]=FP/Sum_L0;
    pTP[i]=TP/Sum_L1;
    pFN[i]=FN/Sum_L1;
    pTN[i]=TN/Sum_L0;   
   
Fig_Num=Fig_Num+1;
plt.figure(Fig_Num) 
plt.plot(pFP,pTP,c='blue')
plt.xlabel("pFA")
plt.ylabel("pTP")
plt.title("ROC curve for Q1 part 2")

plt.show()

## Calculate area under ROC curve
Flip_pTP=np.flip(pTP);
Flip_pFP=np.flip(pFP);
Area_Q1_P2=0;
for i in range(1,SZ):
    Area_Q1_P2= Area_Q1_P2 + Flip_pTP[i]*(Flip_pFP[i]-Flip_pFP[i-1]);
print("Area under ROC curve for Q1 Section A is: "+str(Area_Q1_P2))
print("Finished Q1 Section-A Part-2");

print("Finished Q1 Section-A Part-2");
##------ END Q1 Part 2 ------##

##-------- Q1 Part 3 ---------##
#Calculate  the  probability of error as a function of gamma
P_Err=np.zeros(SZ);
for i in range(0,SZ):
    P_Err[i]=pFP[i]*p[0]+pFN[i]*p[1];
    
Min_P_Err=min(P_Err);

Min_P_Err_ind = [i for i, v in enumerate(P_Err) if v == Min_P_Err];
Best_Gamma=mid_gamma[Min_P_Err_ind];
plt.figure(Fig_Num)
plt.plot(pFP,pTP,c='blue')
plt.xlabel("pFA")
plt.ylabel("pTP")
plt.title("ROC curve for Q1 part 3")
plt.scatter(pFP[Min_P_Err_ind],pTP[Min_P_Err_ind],50,'#d62728')
plt.show()


print("Estimate of the minimum P_Error= "+str(Min_P_Err)+" with threshold value= "+str(Best_Gamma)+" Where pTP is:"+str(pTP[Min_P_Err_ind])+" and pFP is:"+str(pFP[Min_P_Err_ind]));
print("Finished Q1 Section-A Part-3");
##------ END Q1 Part 3 -------##
##------------------------------END Q1 Section A -------------------------##


##-------------------------------- Q1 Section B ---------------------------##
## Redefine Covariance Matrix ##
B_Sigma_0=np.array([[2,    0,    0,   0],
                    [0,    1,    0,   0],
                    [0,    0,    1,   0],
                    [0,    0,    0,   2]]);

B_Sigma_1=np.array([[1,    0,    0,   0],
                    [0,    2,    0,   0],
                    [0,    0,    1,   0],
                    [0,    0,    0,   3]]);
##----------------------------##
B_discriminantScore=np.zeros(N);

for i in range(0,N):
     B_discriminantScore[i] = (evalGaussian(X[:,i],Mu_1[:],B_Sigma_1[:,:])/evalGaussian(X[:,i],Mu_0[:],B_Sigma_0[:,:]));
     # If discriminantScore > gamma, D=1, otherwise, D=0
#- END Calculate Discriminant Score --#
B_gamma = np.sort(B_discriminantScore);
eps=1;
B_mid_gamma=np.zeros(N);
B_mid_gamma[0]=B_gamma[0]-eps;
B_mid_gamma[N-1]=B_gamma[N-1]+eps;
for i in range(0,N-1):
    B_mid_gamma[i]=(B_gamma[i]+B_gamma[i+1])/2;

B_SZ= np.size(B_mid_gamma);
B_pTN=np.zeros(B_SZ);
B_pFN=np.zeros(B_SZ);
B_pTP=np.zeros(B_SZ);
B_pFP=np.zeros(B_SZ);

for i in range(0,B_SZ):
    B_Decision=np.zeros(N);
    B_TP = np.zeros(1); #Number of Correct Decisions/ True Positive
    B_FP =np.zeros(1); #Number of False Alarms/False Positive
    B_TN=np.zeros(1); #Number of True Negatives
    B_FN=np.zeros(1); #Number of False Negatives
    for j in range(0,N):
        B_Decision[j]=(B_discriminantScore[j]>=B_mid_gamma[i]);
        if B_Decision[j]==1 and label[j]==1:
            B_TP = B_TP+1;
        elif  B_Decision[j]==1 and label[j]==0:
             B_FP=B_FP+1;
        elif  B_Decision[j]==0 and label[j]==0:
             B_TN=B_TN+1;
        else:  
             B_FN=B_FN+1;

    B_pFP[i]=B_FP/Sum_L0;
    B_pTP[i]=B_TP/Sum_L1;
    B_pFN[i]=B_FN/Sum_L1;
    B_pTN[i]=B_TN/Sum_L0;
    
Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.plot(B_pFP,B_pTP,c='red')
plt.xlabel("pFA")
plt.ylabel("pTP")
plt.title("ROC curve for Q1 Section B")
plt.show()

#Calculate  the  probability of error as a function of gamma
B_P_Err=np.zeros(B_SZ);
for i in range(0,B_SZ):
    B_P_Err[i]=B_pFP[i]*p[0]+B_pFN[i]*p[1];
    
B_Min_P_Err=min(B_P_Err);

B_Min_P_Err_ind = [i for i, v in enumerate(B_P_Err) if v == B_Min_P_Err];
B_Best_Gamma=B_mid_gamma[B_Min_P_Err_ind];

plt.figure(Fig_Num)
plt.plot(B_pFP,B_pTP,c='red')
plt.xlabel("pFA")
plt.ylabel("pTP")
plt.title("ROC curve for Q1 Section B")
plt.scatter(B_pFP[B_Min_P_Err_ind],B_pTP[B_Min_P_Err_ind],50,'#002728')
plt.show()

## Calculate area under ROC curve
Flip_B_pTP=np.flip(B_pTP);
Flip_B_pFP=np.flip(B_pFP);
Area_Q1_SB=0;
for i in range(1,B_SZ):
    Area_Q1_SB= Area_Q1_SB + Flip_B_pTP[i]*(Flip_B_pFP[i]-Flip_B_pFP[i-1]);
print("Area under ROC curve for Q1 Section B is: "+str(Area_Q1_SB))

 
print("Finished Q1 Section-B");
##------------------------------ END Q1 Section B -------------------------##

##-------------------------------- Q1 Section C ---------------------------##
### Estimate Class Conditional pdfs mean and covariance from the sampled data
Est_Mu_0=np.zeros(n);
Est_Sigma_0=np.zeros((n,n));
Est_Mu_1=np.zeros(n);
Est_Sigma_1=np.zeros((n,n));
S_B=np.zeros((n,n));
S_W=np.zeros((n,n));
L0_Index=np.flatnonzero(label == 0);
L1_Index=np.flatnonzero(label == 1);
X_L0=X[:,L0_Index];
X_L1=X[:,L1_Index];
for i in range(0,n):
    Est_Mu_0[i]=np.mean(X_L0[i,:]);
    Est_Mu_1[i]=np.mean(X_L1[i,:]);
Est_Sigma_0=np.cov(X_L0);
Est_Sigma_1=np.cov(X_L1);
### Calculate Between and Within class scatter
for i in range(len(Est_Mu_0)):  
   for j in range(len(Est_Mu_1)):  
       S_B[i][j] = (Est_Mu_0[i]-Est_Mu_1[i]) * (Est_Mu_0[j]-Est_Mu_1[j]) 
S_W=Est_Sigma_0+Est_Sigma_1;
eigvals, eigvecs = eigh(S_B, S_W,eigvals_only=False);

Max_eigval=max(eigvals);
Max_eigval_ind = [i for i, v in enumerate(eigvals) if v == Max_eigval];
Max_eigvec=eigvecs[:,Max_eigval_ind];

### Project X onto w_transpose to obtain y
y = np.zeros(N);
Color = [];
for i in range(0,N):
    y[i]=np.transpose(Max_eigvec)@X[:,i];
    if label[i]==0:
        Color.append('red');
    else:
        Color.append('blue');
       
Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.scatter(np.linspace(0, N-1, N),y,c=Color,s=2)
plt.xlabel("Data Point")
plt.ylabel("Y")

##Calculate pTP pFP pTN pFN
C_gamma = np.sort(y);
eps=1;
C_mid_gamma=np.zeros(N);
C_mid_gamma[0]=C_gamma[0]-eps;
C_mid_gamma[N-1]=C_gamma[N-1]+eps;
for i in range(0,N-1):
    C_mid_gamma[i]=(C_gamma[i]+C_gamma[i+1])/2;

C_SZ= np.size(C_mid_gamma);
C_pTN=np.zeros(C_SZ);
C_pFN=np.zeros(C_SZ);
C_pTP=np.zeros(C_SZ);
C_pFP=np.zeros(C_SZ);

for i in range(0,C_SZ):
    C_Decision=np.zeros(N);
    C_TP = np.zeros(1); #Number of Correct Decisions/ True Positive
    C_FP =np.zeros(1); #Number of False Alarms/False Positive
    C_TN=np.zeros(1); #Number of True Negatives
    C_FN=np.zeros(1); #Number of False Negatives
    for j in range(0,N):
        C_Decision[j]=(y[j]>=C_mid_gamma[i]);
        if C_Decision[j]==1 and label[j]==1:
            C_TP = C_TP+1;
        elif  C_Decision[j]==1 and label[j]==0:
             C_FP=C_FP+1;
        elif  C_Decision[j]==0 and label[j]==0:
             C_TN=C_TN+1;
        else:  
             C_FN=C_FN+1;

    C_pFP[i]=C_FP/Sum_L0;
    C_pTP[i]=C_TP/Sum_L1;
    C_pFN[i]=C_FN/Sum_L1;
    C_pTN[i]=C_TN/Sum_L0;

#Calculate  the  probability of error as a function of gamma
C_P_Err=np.zeros(C_SZ);
for i in range(0,C_SZ):
    C_P_Err[i]=C_pFP[i]*p[0]+C_pFN[i]*p[1];
    
C_Min_P_Err=min(C_P_Err);

C_Min_P_Err_ind = [i for i, v in enumerate(C_P_Err) if v == C_Min_P_Err];
C_Best_Gamma=C_mid_gamma[C_Min_P_Err_ind];

Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.plot(C_pFP,C_pTP,c='green')
plt.xlabel("pFA")
plt.ylabel("pTP")
plt.title("ROC curve for Q1 Section C")
plt.scatter(C_pFP[C_Min_P_Err_ind],C_pTP[C_Min_P_Err_ind],50,'#d60028')
plt.show()

## Calculate area under ROC curve
Flip_C_pTP=np.flip(C_pTP);
Flip_C_pFP=np.flip(C_pFP);
Area_Q1_SC=0;
for i in range(1,C_SZ):
    Area_Q1_SC= Area_Q1_SC + Flip_C_pTP[i]*(Flip_C_pFP[i]-Flip_C_pFP[i-1]);
print("Area under ROC curve for Q1 Section C is: "+str(Area_Q1_SC))

print("Estimate of the minimum P_Error= "+str(C_Min_P_Err)+" with threshold value= "+str(C_Best_Gamma)+" Where pTP is:"+str(C_pTP[C_Min_P_Err_ind])+" and pFP is:"+str(C_pFP[C_Min_P_Err_ind]));

print("Finished Q1 Section-C");


Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.plot(pFP,pTP,c='blue')
plt.plot(B_pFP,B_pTP,c='red')
plt.plot(C_pFP,C_pTP,c='green')
#plt.legend(['Q1-A','Q1-B','Q1-C'])
plt.scatter(B_pFP[B_Min_P_Err_ind],B_pTP[B_Min_P_Err_ind],50,'#002728')
plt.scatter(pFP[Min_P_Err_ind],pTP[Min_P_Err_ind],50,'#d62728')
plt.scatter(C_pFP[C_Min_P_Err_ind],C_pTP[C_Min_P_Err_ind],50,'#d60028')

plt.xlabel("pFA")
plt.ylabel("pTP")
