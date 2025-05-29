# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 08:11:26 2020

@author: Mahdi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelBinarizer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


################################################################
#                          Question 1                          #
################################################################ 

#------ Generate Data------#
C=4 # Number of Classes
p=[0.25, 0.25, 0.25, 0.25] #Uniform Class Priors
n=3  #Number of dimensions
N_Training=[100, 200, 500, 1000, 2000, 5000] # Training Datset sample sizes
N_Testing=1000 # Testing Dataset sample size

## Class means and covariance ##
Sigma=np.array([[0.5,       0.12,       0.1],
                [  0.1,     0.5,       0],
                [  0,       0.05,     0.5]]);
Mu=np.zeros((C,n));
Mu[0,:]=np.array([-1,  1, 0]);
Mu[1,:]=np.array([-1, -1, 0]);
Mu[2,:]=np.array([ 1, -1, 0]);
Mu[3,:]=np.array([ 1,  1, 0]);
##----------------------------##

## Generate training samples based on class conditional pdfs ## 
N_Train=N_Training[0]
N_random_numbers=np.random.random(N_Train);
label=np.zeros(N_Train);
for i in range(0,N_Train):
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

X = np.zeros((n, N_Train))
L_Counter=np.zeros(C);

Data_Set_L=np.zeros((C,N_Train,n))
for i in range(0,C):
    temp_Mu=np.zeros(n)      
    temp_Mu[:]=Mu[i,:]
    Data_Set_L[i,0:int(Sum_CL[i]),:]= np.random.multivariate_normal(temp_Mu,Sigma,int(Sum_CL[i]));

for i in range(0,N_Train):
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
ax.set_title("Training Dataset Size: "+str(N_Train))


N_random_numbers=np.random.random(N_Testing);
label_Testing=np.zeros(N_Testing);
for i in range(0,N_Testing):
    if N_random_numbers[i]<=p[0]:
        label_Testing[i]=1;
    elif N_random_numbers[i]<=(p[0]+p[1]):
            label_Testing[i]=2;
    elif N_random_numbers[i]<=(p[0]+p[1]+p[2]):
            label_Testing[i]=3;
    else:
        label_Testing[i]=4;
Sum_CL_Testing=np.zeros(C)  
for i in range(0,C):      
    Sum_CL_Testing[i]=sum(label_Testing==(i+1)); #Number of data points with the given class label

X_Testing = np.zeros((n, N_Testing))
L_Counter_Testing=np.zeros(C);

Data_Set_L_Testing=np.zeros((C,N_Testing,n))
for i in range(0,C):
    temp_Mu=np.zeros(n)      
    temp_Mu[:]=Mu[i,:]
    Data_Set_L_Testing[i,0:int(Sum_CL_Testing[i]),:]= np.random.multivariate_normal(temp_Mu,Sigma,int(Sum_CL_Testing[i]));

for i in range(0,N_Testing):
    for CLASS in range(0,C):
        if label_Testing[i]==CLASS+1:
            X_Testing[:,i]= np.transpose(Data_Set_L_Testing[CLASS,int(L_Counter_Testing[CLASS]),:]);
            L_Counter_Testing[CLASS]=L_Counter_Testing[CLASS]+1;
        
Fig_Num=Fig_Num+1

fig=plt.figure(Fig_Num)
ax = fig.add_subplot(111, projection='3d')

m=['o','s','*','d']

ax.scatter(Data_Set_L_Testing[0,:,0], Data_Set_L_Testing[0,:,1], Data_Set_L_Testing[0,:,2], c='#8c564b', marker=m[0], s=55, alpha=1)
ax.scatter(Data_Set_L_Testing[1,:,0], Data_Set_L_Testing[1,:,1], Data_Set_L_Testing[1,:,2], c='#2ca02c', marker=m[1], s=55, alpha=1)
ax.scatter(Data_Set_L_Testing[2,:,0], Data_Set_L_Testing[2,:,1], Data_Set_L_Testing[2,:,2], c='#17becf', marker=m[2], s=55, alpha=1)
ax.scatter(Data_Set_L_Testing[3,:,0], Data_Set_L_Testing[3,:,1], Data_Set_L_Testing[3,:,2], c='#e377c2', marker=m[3], s=55, alpha=1)
ax.legend(['Class 1', 'Class 2', 'Class 3', 'Class 4'])
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title("Testing Dataset size: "+str(N_Testing))
#--- END Generate Data-----#

#------ Theoretically Optimal Classifier ------#

#Define Loss Matrix
lossMatrix= np.ones(C)-np.identity(C);

#---- Calculate Expected Risk and decide on the minimum one ----#
def evalGaussian(x, Mu, Sigma):
    g=multivariate_normal.pdf(x,Mu,Sigma);
    return g

pxgivenL=np.zeros((C,N_Testing))
Class_Posterior=np.zeros((C,N_Testing))
Expected_Risk=np.zeros((C,N_Testing))
Decision=np.zeros(N_Testing)
for i in range(0,C):
    for j in range(0,N_Testing):
        pxgivenL[i,j] = evalGaussian(X_Testing[:,j],Mu[i,:],Sigma[:,:]); #Evaluate p(x|L=i)         !!DOES THIS NEED TO add to one per column?
    Class_Posterior[i,:]=pxgivenL[i,:]*p[i]; #This is not really class posterior, we must divide by p(x) to get class posterior

for j in range(0,N_Testing):
    Expected_Risk[:,j]=lossMatrix@Class_Posterior[:,j];
    Decision[j]= np.argmin(Expected_Risk[:,j])+1;
#- END Calculate Expected Risk --#

##Calculate Confusion Matrix
Confusion_Matrix=np.zeros((C,C));
Sum_DL=np.zeros((C,C));
for D in range(0,C):
        for L in range(0,C):
            Sum_DL[D,L]=sum(np.logical_and(Decision==D+1, label_Testing==L+1)); #Number of data points with Decision=D and label=L
            Confusion_Matrix[D,L]=Sum_DL[D,L]/Sum_CL_Testing[L]; #L-1 since python indexing starts at 0

print("Q1 - The Confusion Matrix is: "+str(Confusion_Matrix))

Color=[]
COLOR_L1=[]
COLOR_L2=[]
COLOR_L3=[]
COLOR_L4=[]

Indeces_L1=np.zeros(int(Sum_CL_Testing[0]),dtype=int); #Contains the index numbers of Dataset X that has label 1
Indeces_L2=np.zeros(int(Sum_CL_Testing[1]),dtype=int);
Indeces_L3=np.zeros(int(Sum_CL_Testing[2]),dtype=int);
Indeces_L4=np.zeros(int(Sum_CL_Testing[3]),dtype=int);
L_Counter=np.zeros(C,dtype=int);
for i in range(0,N_Testing):
    if Decision[i]==label_Testing[i]:
        Color.append('green');
    else:
        Color.append('red');
    if label_Testing[i]==1:
        Indeces_L1[L_Counter[0]]=i;
        L_Counter[0]=L_Counter[0]+1;
        COLOR_L1.append(Color[i]);
    elif label_Testing[i]==2:
        Indeces_L2[L_Counter[1]]=i;
        L_Counter[1]=L_Counter[1]+1;
        COLOR_L2.append(Color[i]);
    elif label_Testing[i]==3:
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
plt.scatter(X_Testing[0,Indeces_L1],X_Testing[1,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L2],X_Testing[1,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L3],X_Testing[1,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L4],X_Testing[1,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.title('Q1 - Green = Correct | Red = Incorrect - Classification of all 4 classes')
plt.xlabel('X1',fontsize=12)
plt.ylabel('X2',fontsize=12)
plt.axis('equal')
    
plt.subplot(312)
plt.scatter(X_Testing[0,Indeces_L1],X_Testing[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L2],X_Testing[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L3],X_Testing[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X_Testing[0,Indeces_L4],X_Testing[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
plt.xlabel('X1',fontsize=12)
plt.ylabel('X3',fontsize=12)
plt.axis('equal')
    
plt.subplot(313)
plt.scatter(X_Testing[1,Indeces_L1],X_Testing[2,Indeces_L1],facecolors='none', edgecolors=COLOR_L1,marker=m[0], s=25, alpha=0.7)
plt.scatter(X_Testing[1,Indeces_L2],X_Testing[2,Indeces_L2],facecolors='none', edgecolors=COLOR_L2,marker=m[1], s=25, alpha=0.7)
plt.scatter(X_Testing[1,Indeces_L3],X_Testing[2,Indeces_L3],c=COLOR_L3,marker=m[2], s=25, alpha=0.7)
plt.scatter(X_Testing[1,Indeces_L4],X_Testing[2,Indeces_L4],facecolors='none', edgecolors=COLOR_L4,marker=m[3], s=25, alpha=0.7)
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
    plt.scatter(X_Testing[0,Scat_Indeces],X_Testing[1,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X2',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+1)
    plt.title("Q1 - Green = Correct | Red = Incorrect - Classification for Class Label= "+str(int(i+1)))
    plt.scatter(X_Testing[0,Scat_Indeces],X_Testing[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X1',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')
    
    plt.subplot(4,3,SUBPLOT+2)
    plt.scatter(X_Testing[1,Scat_Indeces],X_Testing[2,Scat_Indeces],facecolors='none', edgecolors=Scat_Color,marker=Marker, s=15, alpha=0.7)
    plt.xlabel('X2',fontsize=12)
    plt.ylabel('X3',fontsize=12)
    plt.axis('equal')

##Probability of Error
P_Error = (len([i for i, v in enumerate(Color) if v == 'red']))/N_Testing;
print("Minimum Probability of error using the theoretically optimal classifier is: "+str(P_Error))
#--- END Theoretically Optimal Classifier -----#





#----- Use TensorFlow to Train 2-layer MLP ------#

## Changing labels to one-hot encoded vector
lb = LabelBinarizer()
y_train = lb.fit_transform(label)
y_test = lb.transform(label_Testing)
print('Train labels dimension:');print(y_train.shape)
print('Test labels dimension:');print(y_test.shape)

X_train=np.transpose(X)
X_test=np.transpose(X_Testing)

print('Train dimension:');print(X_train.shape)
print('Test dimension:');print(X_test.shape)

from sklearn.metrics import roc_auc_score, accuracy_score
s = tf.InteractiveSession()

## Defining various initialization parameters for 784-512-256-10 MLP model
num_classes = C
num_features = n
num_output = C
num_layers_0 = 5
starter_learning_rate = 0.001
regularizer_rate = 0.1

# Placeholders for the input data
input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
bias_1 = tf.Variable(tf.random_normal([num_output]))

## Initializing weigths and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

## Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
        + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)))
        
## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
## Adam optimzer for finding the right weight
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])
## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Training parameters
batch_size = 4
epochs=20 #Too high epoch will overfit the training data

training_accuracy = []
training_loss = []
testing_accuracy = []

s.run(tf.global_variables_initializer())
for epoch in range(epochs):    
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0],batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                          input_y: y_train[arr[index:index+batch_size]]})
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                         input_y: y_train}))
    training_loss.append(s.run(loss, {input_X: X_train, 
                                      input_y: y_train}))
    
    ## Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1), 
                            s.run(predicted_y, {input_X: X_test}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch],
                                                                   testing_accuracy[epoch]))

