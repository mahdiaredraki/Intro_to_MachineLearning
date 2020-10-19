# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:57:04 2020

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
n= 3; #Number of dimensions
N= 100; #Number of samples simulated
C= 4; #Number of classes
##----------------------------------##

## Class means and covariance ##
Sigma=np.array([[1.05,       0,       0],
                [  0,     1.05,       0],
                [  0,       0,     1.05]]);
Mu=np.zeros((C,n));
Mu[0,:]=np.array([-1,  1, 0]);
Mu[1,:]=np.array([-1, -1, 0]);
Mu[2,:]=np.array([ 1, -1, 0]);
Mu[3,:]=np.array([ 1,  1, 0]);
##----------------------------##

## Class priors ##
p=np.array([0.2, 0.25, 0.25, 0.3]); #p[0]=class 1 prior, p[1]=class 2 prior, p[2]=class 3 prior, p[3]=class 4 prior
##--------------##

## Generate samples based on class conditional pdfs ## !!! ONLY RUN ONCE AND THEN COMMENT OUT
N_random_numbers=np.random.random(N);
label=np.zeros(N);
for i in range(0,N):
    if N_random_numbers[i]<=p[0]:
        label[i]=1;
    elif N_random_numbers[i]<=(p[0]+p[1]):
            label[i]=2;
    elif N_random_numbers[i]<=(p[0]+p[1]+p[2]):
            label[i]=3;
    else:
        label[i]=4;
Sum_CL=np.zeros(C)  
for i in range(0,C):      
    Sum_CL[i]=sum(label==(i+1)); #Number of data points with the given class label

X = np.zeros((n, N))
L_Counter=np.zeros(C);

Data_Set_L=np.zeros((C,N,n))
for i in range(0,C):
    temp_Mu=np.zeros(n)      
    temp_Mu[:]=Mu[i,:]
    Data_Set_L[i,0:int(Sum_CL[i]),:]= np.random.multivariate_normal(temp_Mu,Sigma,int(Sum_CL[i]));

for i in range(0,N):
    for CLASS in range(0,C):
        if label[i]==CLASS+1:
            X[:,i]= np.transpose(Data_Set_L[CLASS,int(L_Counter[CLASS]),:]);
            L_Counter[CLASS]=L_Counter[CLASS]+1;
        
Fig_Num=1;

fig=plt.figure(Fig_Num)
ax = fig.add_subplot(111, projection='3d')

m=['o','s','*','d']

ax.scatter(Data_Set_L[0,:,0], Data_Set_L[0,:,1], Data_Set_L[0,:,2], c='#8c564b', marker=m[0], s=55, alpha=1)
ax.scatter(Data_Set_L[1,:,0], Data_Set_L[1,:,1], Data_Set_L[1,:,2], c='#2ca02c', marker=m[1], s=55, alpha=1)
ax.scatter(Data_Set_L[2,:,0], Data_Set_L[2,:,1], Data_Set_L[2,:,2], c='#17becf', marker=m[2], s=55, alpha=1)
ax.scatter(Data_Set_L[3,:,0], Data_Set_L[3,:,1], Data_Set_L[3,:,2], c='#e377c2', marker=m[3], s=55, alpha=1)
ax.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4'])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

#---------- END Setup the Given vectors and matrices --------#

#Define Loss Matrix
lossMatrix= np.ones(C)-np.identity(C);

#---- Calculate Expected Risk and decide on the minimum one ----#
def evalGaussian(x, Mu, Sigma):
    g=multivariate_normal.pdf(x,Mu,Sigma);
    return g

pxgivenL=np.zeros((C,N))
Class_Posterior=np.zeros((C,N))
Expected_Risk=np.zeros((C,N))
Decision=np.zeros(N)
for i in range(0,C):
    for j in range(0,N):
        pxgivenL[i,j] = evalGaussian(X[:,j],Mu[i,:],Sigma[:,:]); #Evaluate p(x|L=i)         !!DOES THIS NEED TO add to one per column?
    Class_Posterior[i,:]=pxgivenL[i,:]*p[i]; #This is not really class posterior, we must divide by p(x) to get class posterior

for j in range(0,N):
    Expected_Risk[:,j]=lossMatrix@Class_Posterior[:,j];
    Decision[j]= np.argmin(Expected_Risk[:,j])+1;
#- END Calculate Expected Risk --#

##Calculate Confusion Matrix
Confusion_Matrix=np.zeros((C,C));
Sum_DL=np.zeros((C,C));
for D in range(0,C):
        for L in range(0,C):
            Sum_DL[D,L]=sum(np.logical_and(Decision==D+1, label==L+1)); #Number of data points with Decision=D and label=L
            Confusion_Matrix[D,L]=Sum_DL[D,L]/Sum_CL[L]; #L-1 since python indexing starts at 0

print("Q2 Part 2 - The Confusion Matrix is: "+str(Confusion_Matrix))
print("Finished Q2 part 2")

Color=[]
COLOR_L1=[]
COLOR_L2=[]
COLOR_L3=[]
COLOR_L4=[]

Indeces_L1=np.zeros(int(Sum_CL[0]),dtype=int); #Contains the index numbers of Dataset X that has label 1
Indeces_L2=np.zeros(int(Sum_CL[1]),dtype=int);
Indeces_L3=np.zeros(int(Sum_CL[2]),dtype=int);
Indeces_L4=np.zeros(int(Sum_CL[3]),dtype=int);
L_Counter=np.zeros(C,dtype=int);
for i in range(0,N):
    if Decision[i]==label[i]:
        Color.append('green');
    else:
        Color.append('red');
    if label[i]==1:
        Indeces_L1[L_Counter[0]]=i;
        L_Counter[0]=L_Counter[0]+1;
        COLOR_L1.append(Color[i]);
    elif label[i]==2:
        Indeces_L2[L_Counter[1]]=i;
        L_Counter[1]=L_Counter[1]+1;
        COLOR_L2.append(Color[i]);
    elif label[i]==3:
        Indeces_L3[L_Counter[2]]=i;
        L_Counter[2]=L_Counter[2]+1;
        COLOR_L3.append(Color[i]);
    else:
        Indeces_L4[L_Counter[3]]=i;
        L_Counter[3]=L_Counter[3]+1;
        COLOR_L4.append(Color[i]);
            
Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)     
plt.subplot(311)
plt.scatter(X[0,Indeces_L1],X[1,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L2],X[1,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L3],X[1,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L4],X[1,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.title('Q2-Part3 - Green = Correct | Red = Incorrect - Classification of all 4 classes')
plt.xlabel('X1',fontsize=12)
plt.ylabel('X2',fontsize=12)
plt.axis('equal')
    
plt.subplot(312)
plt.scatter(X[0,Indeces_L1],X[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L2],X[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L3],X[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L4],X[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.xlabel('X1',fontsize=12)
plt.ylabel('X3',fontsize=12)
plt.axis('equal')
    
plt.subplot(313)
plt.scatter(X[1,Indeces_L1],X[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L2],X[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L3],X[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L4],X[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.xlabel('X2',fontsize=12)
plt.ylabel('X3',fontsize=12)
plt.axis('equal')

Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.subplots_adjust(hspace=1.2, wspace=0.8)
for i in range(0,C):
    if i==0:
        SUBPLOT=1;
        Scat_Color=COLOR_L1
        Scat_Indeces=Indeces_L1
        Marker=m[0]
    elif i==1:
        SUBPLOT=4;
        Scat_Color=COLOR_L2
        Scat_Indeces=Indeces_L2
        Marker=m[1]
    elif i==2:
        SUBPLOT=7;
        Scat_Color=COLOR_L3
        Scat_Indeces=Indeces_L3
        Marker=m[2]
    else:
        SUBPLOT=10;
        Scat_Color=COLOR_L4
        Scat_Indeces=Indeces_L4
        Marker=m[3]
        
    plt.subplot(4,3,SUBPLOT)
    plt.scatter(X[0,Scat_Indeces],X[1,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X2',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+1)
    plt.title("Q2-Part3 - Green = Correct | Red = Incorrect - Classification for Class Label= "+str(int(i+1)))
    plt.scatter(X[0,Scat_Indeces],X[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+2)
    plt.scatter(X[1,Scat_Indeces],X[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X2',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')
    

print("Finished Q2 part 3")

#Define Loss Matrix
lossMatrix=np.array([[  0,     1,   2,    3],
                     [ 10,     0,   5,   10],
                     [ 20,    10,   0,    1],
                     [ 30,    20,   1,    0]]);

#---- Calculate Expected Risk and decide on the minimum one ----#
def evalGaussian(x, Mu, Sigma):
    g=multivariate_normal.pdf(x,Mu,Sigma);
    return g

pxgivenL=np.zeros((C,N))
Class_Posterior=np.zeros((C,N))
Expected_Risk=np.zeros((C,N))
Decision=np.zeros(N)
for i in range(0,C):
    for j in range(0,N):
        pxgivenL[i,j] = evalGaussian(X[:,j],Mu[i,:],Sigma[:,:]); #Evaluate p(x|L=i)         !!DOES THIS NEED TO add to one per column?
    Class_Posterior[i,:]=pxgivenL[i,:]*p[i]; #This is not really class posterior, we must divide by p(x) to get class posterior

for j in range(0,N):
    Expected_Risk[:,j]=lossMatrix@Class_Posterior[:,j];
    Decision[j]= np.argmin(Expected_Risk[:,j])+1;
#- END Calculate Expected Risk --#

##Calculate Confusion Matrix
Confusion_Matrix=np.zeros((C,C));
Sum_DL=np.zeros((C,C));
for D in range(0,C):
        for L in range(0,C):
            Sum_DL[D,L]=sum(np.logical_and(Decision==D+1, label==L+1)); #Number of data points with Decision=D and label=L
            Confusion_Matrix[D,L]=Sum_DL[D,L]/Sum_CL[L]; #L-1 since python indexing starts at 0

print("Q2 Part B - The Confusion Matrix is: "+str(Confusion_Matrix))


Color=[]
COLOR_L1=[]
COLOR_L2=[]
COLOR_L3=[]
COLOR_L4=[]

Indeces_L1=np.zeros(int(Sum_CL[0]),dtype=int); #Contains the index numbers of Dataset X that has label 1
Indeces_L2=np.zeros(int(Sum_CL[1]),dtype=int);
Indeces_L3=np.zeros(int(Sum_CL[2]),dtype=int);
Indeces_L4=np.zeros(int(Sum_CL[3]),dtype=int);
L_Counter=np.zeros(C,dtype=int);
for i in range(0,N):
    if Decision[i]==label[i]:
        Color.append('green');
    else:
        Color.append('red');
    if label[i]==1:
        Indeces_L1[L_Counter[0]]=i;
        L_Counter[0]=L_Counter[0]+1;
        COLOR_L1.append(Color[i]);
    elif label[i]==2:
        Indeces_L2[L_Counter[1]]=i;
        L_Counter[1]=L_Counter[1]+1;
        COLOR_L2.append(Color[i]);
    elif label[i]==3:
        Indeces_L3[L_Counter[2]]=i;
        L_Counter[2]=L_Counter[2]+1;
        COLOR_L3.append(Color[i]);
    else:
        Indeces_L4[L_Counter[3]]=i;
        L_Counter[3]=L_Counter[3]+1;
        COLOR_L4.append(Color[i]);
            
Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)     
plt.subplot(311)
plt.scatter(X[0,Indeces_L1],X[1,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L2],X[1,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L3],X[1,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L4],X[1,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.title('Q2 Part B - Classification of all 4 classes')
plt.xlabel('X1',fontsize=12)
plt.ylabel('X2',fontsize=12)
plt.axis('equal')
    
plt.subplot(312)
plt.scatter(X[0,Indeces_L1],X[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L2],X[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L3],X[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[0,Indeces_L4],X[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.xlabel('X1',fontsize=12)
plt.ylabel('X3',fontsize=12)
plt.axis('equal')
    
plt.subplot(313)
plt.scatter(X[1,Indeces_L1],X[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L2],X[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L3],X[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X[1,Indeces_L4],X[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.xlabel('X2',fontsize=12)
plt.ylabel('X3',fontsize=12)
plt.axis('equal')

Fig_Num=Fig_Num+1;
plt.figure(Fig_Num)
plt.subplots_adjust(hspace=1.2, wspace=0.8)
for i in range(0,C):
    if i==0:
        SUBPLOT=1;
        Scat_Color=COLOR_L1
        Scat_Indeces=Indeces_L1
        Marker=m[0]
    elif i==1:
        SUBPLOT=4;
        Scat_Color=COLOR_L2
        Scat_Indeces=Indeces_L2
        Marker=m[1]
    elif i==2:
        SUBPLOT=7;
        Scat_Color=COLOR_L3
        Scat_Indeces=Indeces_L3
        Marker=m[2]
    else:
        SUBPLOT=10;
        Scat_Color=COLOR_L4
        Scat_Indeces=Indeces_L4
        Marker=m[3]
        
    plt.subplot(4,3,SUBPLOT)
    plt.scatter(X[0,Scat_Indeces],X[1,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X2',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+1)
    plt.title("Q2 Part B - Classification for Class Label= "+str(int(i+1)))
    plt.scatter(X[0,Scat_Indeces],X[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+2)
    plt.scatter(X[1,Scat_Indeces],X[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X2',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')
    
print("Finished Q2 part B")





