import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1/(1+np.exp(-1*z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_label = np.zeros((50000,))
    validation_label = np.zeros((10000,))
    test_label = np.zeros((10000,))
    for i in range(0,10):
      if i==0:
        train=mat.get("train"+str(i))
        test=mat.get("test"+str(i))              
        np.random.shuffle(train)
        validation_data=train[:1000,:]
        train_data=train[1000:,:]
        test_data=test
        train_label[:train_data.shape[0]]=i
        test_label[:test_data.shape[0]]=i
        validation_label[:validation_data.shape[0]]=i
      else:
        train=mat.get("train"+str(i))
        test=mat.get("test"+str(i))        
        np.random.shuffle(train)
        val_len=validation_data.shape[0]
        validation_data=np.vstack((validation_data,train[:1000,:]))
        
        train_len=train_data.shape[0]
        test_len=test_data.shape[0]
                
        train_data=np.vstack((train_data,train[1000:,:]))
        test_data=np.vstack((test_data,test))
        train_label[train_len:train_data.shape[0]]=i
        test_label[test_len:test_data.shape[0]]=i
        validation_label[val_len:validation_data.shape[0]]=i  

    # Feature selection
    # Your code here.
    train_data=train_data/255
    test_data=test_data/255
    validation_data=validation_data/255

    cols=train_data.shape[1]
    train_same_col=[]
    test_same_col=[]
    valid_same_col=[]
    for i in range(cols):
      for j in range(i+1,cols):
        if (train_data[:,i]==(train_data[:,j])).all():
          if i not in train_same_col:
            train_same_col.append(i)
        if (test_data[:,i]==(test_data[:,j])).all():
          if i not in test_same_col:
            test_same_col.append(i)
        if (validation_data[:,i]==(validation_data[:,j])).all():
          if i not in valid_same_col:
            valid_same_col.append(i)
    
    train_indices=list(set(np.arange(cols)).difference(set(train_same_col)))
    test_indices=list(set(np.arange(cols)).difference(set(test_same_col)))
    val_indices=list(set(np.arange(cols)).difference(set(valid_same_col)))          

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

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
    yl = np.zeros((n,10))
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


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n=data.shape[0]
    data=np.hstack((data,np.ones((n,1),int)))
    a=sigmoid(np.dot(data,w1.T))    
    a_b = np.hstack((a, np.ones((a.shape[0], 1),int)))
    labels = sigmoid(np.dot(a_b,w2.T))    
    labels = np.argmax(labels, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
