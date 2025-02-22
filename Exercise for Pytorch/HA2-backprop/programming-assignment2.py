"""
         Programming Assignment 2
         
To be submitted via canvas, just as Programming Assignment 1

This program builds a two-layer neural network for the Iris dataset.
The first layer is a relu layer with 10 units, and the second one is 
a softmax layer. The network structure is specified in the "train" function.

The parameters are learned using SGD.  The forward propagation and backward 
propagation are carried out in the "compute_neural_net_loss" function.  The codes
for the propagations are deleted.  Your task is to fill in the missing codes.

"""

# In this exercise, we are going to work with a two-layer neural network
# first layer is a relu layer with 10 units, and second one is a softmax layer.
# randomly initialize parameters
    

import numpy as np
import os, sys
import math

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

def get_data():
    # Load datasets.
    train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
        dtype=float, delimiter=',') 
    test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
        dtype=float, delimiter=',') 
    train_x = train_data[:, :4]
    train_y = train_data[:, 4].astype(np.int64)
    test_x = test_data[:, :4]
    test_y = test_data[:, 4].astype(np.int64)

    ''' 切片操作说明
    arr[start_row:stop_row:step_row, start_col:stop_col:step_col]
    '''

    return train_x, train_y, test_x, test_y

def compute_neural_net_loss(params, X, y, reg=0.0):
    """
    Neural network loss function.
    Inputs:
    - params: dictionary of parameters, including "W1", "b1", "W2", "b2"
    - X: N x D array of training data. Each row is a D-dimensional point.
    - y: 1-d array of shape (N, ) for the training labels.

    Returns:
    - loss: the softmax loss with regularization
    - grads: dictionary of gradients for the parameters in params
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    loss = 0.0
    grads = {}

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    z1 = np.dot(X, W1) + b1  
    relu = lambda x: x * (x > 0)
    h = relu(z1)

    # 隐藏层到输出层
    z2 = np.dot(h, W2) + b2  

    y_pred = np.exp(z2)
    y_pred /= np.sum(y_pred, axis=1, keepdims=True)

    data_loss = -np.sum(np.log(y_pred[np.arange(N), y])) / N
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = data_loss + reg_loss

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # 反向传播
    # 输出层梯度
    dz2 = y_pred.copy()
    dz2[np.arange(N), y] -= 1
    dz2 /= N

    # 计算 W2 和 b2 的梯度
    dW2 = np.dot(h.T, dz2) + reg * W2  
    db2 = np.sum(dz2, axis=0)

    # 计算隐藏层梯度
    dh = np.dot(dz2, W2.T)  
    dz1 = dh * (z1 > 0)

    # 计算 W1 和 b1 的梯度
    dW1 = np.dot(X.T, dz1) + reg * W1  
    db1 = np.sum(dz1, axis=0)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    grads['W1']=dW1
    grads['W2']=dW2
    grads['b1']=db1
    grads['b2']=db2
    
    return loss, grads

def predict(params, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - params: dictionary of parameters, including "W1", "b1", "W2", "b2"
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    y_pred = np.zeros(X.shape[1]) # 1?0?
   
    relu = lambda x: x * (x > 0)
    z1 = np.dot(X,W1)+b1
    u1 = relu(z1)
    z2 = np.dot(u1,W2)+b2
    y_pred = np.argmax(z2, axis=1)
    
    return y_pred

def acc(ylabel, y_pred):
    return np.mean(ylabel == y_pred)

def sgd_update(params, grads, learning_rate):
    """
    Perform sgd update for parameters in params.
    """
    for key in params:
        params[key] += -learning_rate * grads[key]

def validate_gradient():
    """
    Function to validate the implementation of gradient computation.
    Should be used together with gradient_check.py.
    This is a useful thing to do when you implement your own gradient
    calculation methods.
    It is not required for this assignment.
    """
    from gradient_check import eval_numerical_gradient, rel_error
    # randomly initialize W
    dim = 4
    num_classes = 4
    num_inputs = 5
    params = {}
    std = 0.001
    params['W1'] = std * np.random.randn(dim, 10)
    params['b1'] = np.zeros(10)
    params['W2'] = std * np.random.randn(10, num_classes)
    params['b2'] = np.zeros(num_classes)

    X = np.random.randn(num_inputs, dim)
    y = np.array([0, 1, 2, 2, 1])

    loss, grads = compute_neural_net_loss(params, X, y, reg=0.1)
    # these should all be less than 1e-8 or so
    for param_name in params:
      f = lambda W: compute_neural_net_loss(params, X, y, reg=0.1)[0]
      param_grad_num = eval_numerical_gradient(f, params[param_name], verbose=False)
      print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

def train(X, y, Xtest, ytest, learning_rate=1e-3, reg=1e-5, epochs=100, batch_size=20):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    num_iters_per_epoch = int(math.floor(1.0*num_train/batch_size))
    
    # In this exercise, we are going to work with a two-layer neural network
    # first layer is a relu layer with 10 units, and second one is a softmax layer.
    # randomly initialize parameters
    params = {}
    std = 0.001
    params['W1'] = std * np.random.randn(dim, 10)
    params['b1'] = np.zeros(10)
    params['W2'] = std * np.random.randn(10, num_classes)
    params['b2'] = np.zeros(num_classes)

    for epoch in range(max_epochs):
        perm_idx = np.random.permutation(num_train) 
        '''
        在每个 epoch 开始时，使用 np.random.permutation 函数对训练样本的索引进行随机打乱
        '''
        # perform mini-batch SGD update
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it*batch_size:(it+1)*batch_size]
            batch_x = X[idx]
            batch_y = y[idx]
            
            # evaluate loss and gradient
            loss, grads = compute_neural_net_loss(params, batch_x, batch_y, reg)

            # update parameters
            sgd_update(params, grads, learning_rate)
            
        # evaluate and print every 10 steps
        if epoch % 10 == 0:
            train_acc = acc(y, predict(params, X))
            test_acc = acc(ytest, predict(params, Xtest))
            print('Epoch %4d: loss = %.2f, train_acc = %.4f, test_acc = %.4f' \
                % (epoch, loss, train_acc, test_acc))
    
    return params

# validate_gradient()  # don't worry about this.
# sys.exit()

max_epochs = 200
batch_size = 20
learning_rate = 0.1
reg = 0.001

'''参数说明
1. max_epochs
    - max_epochs 指的是训练过程中整个数据集将被模型遍历的最大次数。一个 epoch 代表模型完整地处理了一次所有的训练数据。
    - max_epochs = 200 意味着模型最多会对整个训练数据集进行 200 次遍历和参数更新。
2. batch_size
    - batch_size 表示在每次参数更新时，模型所使用的训练样本数量。
    - 在训练过程中，通常不会一次性使用所有的训练数据来更新模型参数，而是将数据集分成多个小的批次（batch），每次只使用一个批次的数据进行计算和参数更新。
3. epoch与batch
    - 每个epoch要将所有的数据遍历一次用于训练，而一次训练一个batch可能只是一部分数据。
    - 所以每个epoch要多次迭代多个batch，使得能够覆盖所有的数据
'''
# get training and testing data
train_x, train_y, test_x, test_y = get_data()
params = train(train_x, train_y, test_x, test_y, learning_rate, reg, max_epochs, batch_size)

# Classify two new flower samples.
def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
new_x = new_samples()
predictions = predict(params, new_x)

print("New Samples, Class Predictions:    {}\n".format(predictions))
