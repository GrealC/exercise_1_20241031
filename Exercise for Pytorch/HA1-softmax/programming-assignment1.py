"""
           Programming Assignment 1
         

This program learns a softmax model for the Iris dataset (included).
There is a function, compute_softmax_loss, that computes the
softmax loss and the gradient. It is left empty. Your task is to write
the function.
     
"""
import numpy as np
import os
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

    return train_x, train_y, test_x, test_y

def compute_softmax_loss(W, X, y, reg):
    """
    Softmax loss function.
    Inputs:
    - W: D x K array of weight, where K is the number of classes.
    - X: N x D array of training data. Each row is a D-dimensional point.
    - y: 1-d array of shape (N, ) for the training labels.
    - reg: weight regularization coefficient.

    Returns:
    - softmax loss: NLL/N +  0.5 *reg* L2 regularization,
            
    - dW: the gradient for W.
    """


    #############################################################################
    # TODO: Compute the softmax loss and its gradient.                          #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    num_train = X.shape[0]
    # # 计算得分
    scores = X.dot(W)
    # # 为了数值稳定性，减去每行的最大值
    scores -= np.max(scores, axis=1, keepdims=True)
    
    # 计算指数
    exp_scores = np.exp(scores)
    # 计算概率分布
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算负对数似然损失

    '''
    负对数似然损失是交叉熵损失的一种形式，它衡量真实标签和预测概率之间的差异。
    使用 probs[np.arange(num_train), y] 索引出每个样本的真实类别概率。
    取负对数是为了将概率值转换为损失值，概率越小，损失越大。
    '''
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    # 计算平均损失
    data_loss = np.sum(correct_logprobs) / num_train
    # 计算正则化损失
    reg_loss = 0.5 * reg * np.sum(W * W)
    # 计算总损失
    loss = data_loss + reg_loss

    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train
    dW = X.T.dot(dscores)
    dW += reg * W

    return loss, dW

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def predict(W, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: D x K array of weights. K is the number of classes.
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    score = X.dot(W)
    y_pred = np.argmax(score, axis=1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred

def acc(ylabel, y_pred):
    return np.mean(ylabel == y_pred)


def train(X, y, Xtest, ytest, learning_rate=1e-3, reg=1e-5, epochs=100, batch_size=20):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    num_iters_per_epoch = int(math.floor(1.0*num_train/batch_size))
    
    # randomly initialize W
    W = 0.001 * np.random.randn(dim, num_classes)

    for epoch in range(max_epochs):
        perm_idx = np.random.permutation(num_train)
        # perform mini-batch SGD update
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it*batch_size:(it+1)*batch_size]
            batch_x = X[idx]
            batch_y = y[idx]
            
            # evaluate loss and gradient
            loss, grad = compute_softmax_loss(W, batch_x, batch_y, reg)

            # update parameters
            W += -learning_rate * grad
            

        # evaluate and print every 10 steps
        if epoch % 10 == 0:
            train_acc = acc(y, predict(W, X))
            test_acc = acc(ytest, predict(W, Xtest))
            print('Epoch %4d: loss = %.2f, train_acc = %.4f, test_acc = %.4f' \
                % (epoch, loss, train_acc, test_acc))
    
    return W

max_epochs = 200
batch_size = 20
learning_rate = 0.1
reg = 0.01

# get training and testing data
train_x, train_y, test_x, test_y = get_data()
W = train(train_x, train_y, test_x, test_y, learning_rate, reg, max_epochs, batch_size)

# Classify two new flower samples.
def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
new_x = new_samples()
predictions = predict(W, new_x)

print("New Samples, Class Predictions:    {}\n".format(predictions))
