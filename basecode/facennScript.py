'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1/(1+np.exp(-1*z))

    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    b1=np.ones((training_data.shape[0],1),int)
    b2=np.ones((training_data.shape[0],1),int)
    training_data=np.hstack((training_data,b1)) # Bias Added
    Z1=np.dot(training_data,w1.T)
    a1=sigmoid(Z1)

    hidden_data=np.hstack((a1,b2))
    Z2=np.dot(hidden_data,w2.T)
    a2=sigmoid(Z2)

    # 1-of-K coding scheme
    n=training_data.shape[0]
    yl = np.zeros((n,2))
    for i in range (n):
        yl[i][int(training_label[i])]=1

    ln_out_hidden=np.log(a2)
    ln_neg_out_hidden=np.log(1-a2)
    jw=-(np.sum(yl*ln_out_hidden + (1-yl)*ln_neg_out_hidden))/n

    dout=a2-yl
    dout_dw2=np.dot(dout.T,hidden_data)
    dout_dw1 = np.dot(((np.dot(dout,w2))*((1-hidden_data)*hidden_data)).T,training_data)
    # dout_dw1 = np.dot((1-hidden_data)*hidden_data*np.dot(dout,w2),training_data)
    dout_dw1 = dout_dw1[:n_hidden,:]

    reg_jw=jw + (lambdaval/(2*n))*(np.sum(w1**2)+np.sum(w2**2))

    dout_dw1=(dout_dw1+lambdaval*w1)/n
    dout_dw2=(dout_dw2+lambdaval*w2)/n

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((dout_dw1.flatten(), dout_dw2.flatten()),0) 
    obj_val = reg_jw
    return (obj_val, obj_grad)

    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    n=data.shape[0]
    data=np.hstack((data,np.ones((n,1),int)))
    a=sigmoid(np.dot(data,w1.T))    
    a_b = np.hstack((a, np.ones((a.shape[0], 1),int)))
    labels = sigmoid(np.dot(a_b,w2.T))    
    labels = np.argmax(labels, axis=1)

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
