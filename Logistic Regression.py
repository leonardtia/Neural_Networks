# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append('~/PycharmProjects/py36/Machine_Learning')
import Data_Preprocessing as dp

np.random.seed(42)
m, n_x = 500, 3
X = np.random.randn(n_x,m)

r_w = [5,3,4]

r_b=7

pd_X = pd.DataFrame(X)
y=np.zeros((1,m))
pd_y = pd.DataFrame(y)


def build_y(Row):
    y = np.dot(r_w,Row)+r_b
    return y

pd_y = pd_X.apply(build_y,axis=0).reshape(1,m)



def z(w,x,b):
    return np.dot(np.transpose(w),x)+b

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    return 1/(1+np.exp(-z))

y= sigmoid(pd_y)

X_train,X_test,y_train,y_test = dp.get_TrainingSet_and_Test_set(pd_X.T,y.T,0.3)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T




def get_y(Row):
    if Row[0]>0.5:
        return 1
    else:
        return 0
    
y = pd.DataFrame(y).apply(get_y,axis=0).reshape(1,m)

def alpha(w,X,b):
    return sigmoid(z(w,X,b))

def cost(A,Y,m):
    '''
    成本函数
    '''
    return -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    初始化
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    正向传播求成本函数
    反向传播求梯度（w,b求导）
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    A = alpha(w,X,b)
    _cost = cost(A,Y,m)
    dw = 1/m * np.dot(X,np.transpose((A-Y)))
    db = 1/m * np.sum(A-Y)
    
    return _cost,dw,db

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    优化迭代
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):
        _J,_dw,_db = propagate(w,b,X,Y)
        w = w - learning_rate * _dw
        b = b - learning_rate * _db
        if np.remainder(i,100) == 0:
            costs.append(_J)
        if print_cost:
            if np.remainder(i,5000)==0:
                print('Cost after iteration %i:%f'%(i,_J))
            
    params = {'w':w,'b':b}
    grads = {'dw':_dw,'db':_db}
    return params,grads,costs

def predict(w,b,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    拟合并验证test
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    A = alpha(w,X,b)
    _y = pd.DataFrame(A).apply(get_y,axis=0).reshape(1,m)
    return _y



w,b = initialize_with_zeros(n_x)
params,grads,costs = optimize(w,b,X_train,y_train,200000,0.009,True)

print ("w = %s,r_w = %s"%(str(params["w"]),r_w))
print ("b = %s,r_b = %s"%(str(params["b"]),r_b))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

result = predict(params["w"],params["b"],X_test)
m = X_test.shape[1]
rs_y = pd.DataFrame(result).apply(get_y,axis=0).reshape(1,m)
#print("train accuracy: {} %".format(100 - np.mean(np.abs(result - y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(rs_y - y_test)) * 100))


        


