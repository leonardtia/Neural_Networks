#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 11:38:14 2018

@author: leonard_tia
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import Deep_NN_model as dnnm
import Deep_NN_code  as dnnc

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

def getData(datasets):
    for dataset in datasets.keys():
        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])
        if dataset == "blobs":
            Y = Y%2
        yield X,Y
# Visualize the data:
fig = plt.figure(figsize=(8,8))

def visualize(X,Y,i):
    ax = fig.add_subplot(2,2,i)
    ax.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    ax.set_title('X=%s,Y=%s'%(X.shape,Y.shape))

#绘制决策边界
def plot_decision_boundary(model, X, y,mm=''):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # 生成距离为h的点的网格(计算出x的最小值和最大值，y的最小值和最大值，分别用0.01做单位切片，
    #然后其张为XX的矩阵(y-dim,x-dim)和yy(y-dim,x-dim)的矩阵，构成网格)
    #x-dim = np.arange(x_min, x_max, h)
    #y-dim = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 预测整个网格的函数值(np.c_是将xx和yy拉平后，按列组合起来（y-dim*x-dim，2）)然后将这个矩阵
    #作为X传给预测函数，这个X矩阵里，XX和YY分别代表X的两个特征
    if mm == 'Deep_NN':
        Z = model(np.c_[xx.ravel(), yy.ravel()].T,y)
    else:
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        
    #在将拟合后的结果返回还原成之前张成的矩阵大小
    Z = Z.reshape(xx.shape)
    # 绘制轮廓和训练示例，xx,yy是网格，z矩阵是这个网格上的等高线，就是绘制后景
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    #绘制散点图，前景，颜色用y来区分
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

i = 1
for X,Y in getData(datasets):
    visualize(X,Y,i)
    #先用基本的LR进行分类处理
    #clf = sklearn.linear_model.LogisticRegressionCV();
    #clf.fit(X.T, Y.T);
    #绘制决策边界
    #plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    i+=1  
plt.show()

dataset = "blobs"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])
if dataset == "blobs":
    Y = Y%2
layers_dims = [X.shape[0],9,1]
parameters = dnnm.L_layer_model(X, Y, layers_dims, num_iterations = 250000, 
                                print_cost = True)
pred = dnnm.predict(X,Y,parameters)
print ('Accuracy: %d' % float((np.dot(Y,pred.T) + 
                                   np.dot(1-Y,1-pred.T))/float(Y.size)*100) + '%')
plot_decision_boundary(lambda x,y: dnnm.predict(x,y,parameters), X, Y,mm='Deep_NN')
