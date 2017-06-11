'''This project was mediated through Michael Guerzhoy and is not 
to be copy and used for educational purposes which can lead to Academic Integrity'''

# Import all modules from pylab
from pylab import *

# Numpy Modules
import numpy as np
from numpy import random

# Matplotlib Modules
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties

# Scipy modules
from scipy.misc import imread
from scipy.misc import imresize
import scipy.stats
from scipy.ndimage import filters
from scipy.io import loadmat

# Modules for reading data
import urllib
import cPickle
import os
import timeit
import time

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

def get_data(M):
    '''we have a total 10 images
       we first obtain the data images and concatenate vertically stacking them
       The end result will have a height of 60,000'''
    Data = M["train0"]
    for i in range(1,10):
        Data = np.vstack((Data,M["train" + str(i)]))
        
    for i in range(0,10):
        if i == 0:
            y = np.zeros((1,10))
            y[0][i] = 1
            y = repeat(y[0][newaxis,:],len(M["train"+str(i)]),0)
            y_one_hot = y
        else:
            y = np.zeros((1,10))
            y[0][i] = 1
            y = repeat(y[0][newaxis,:],len(M["train"+str(i)]),0)
            y_one_hot = np.vstack((y_one_hot,y))
        
    return Data.T,y_one_hot.T
        
#input x to be of the image from the training set. 
#input already divided into its flattened image vector (len of 784) 

initial_w = np.ones((10,784))/10000
bias = 1

def softmax(X, W):
    '''
    Takes the matrices X and W and returns the softmax probabilities matrix P
    :param X: flattened image matrix where each row is a flattened image with a bias (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :return: the softmax probabilities matrix (nx10 numpy array)
    '''
    O = np.dot(X, W.T)
    P = np.exp(O)/(np.array([np.sum(np.exp(O), 1)])).T
    return P


def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

# y_ = get_data(M)[1]
# y_out = sum_w_bias(get_data(M)[0],initial_w,bias
# y = softmax(y_out)

def NLL(y, y_):
    return -sum(y_*log(y)) 

def gradient(P,X,Y):
    grad = np.dot(X.T,P-Y)
    return grad

def finite_diff(W,x,y,h):
    '''Param W: initial weights of 10x784. When computing the gradient we fix all the 
       values except one of them.
       Cost is dependant on softmax which is dependant on the output where we will change
       the weights.'''
    grad = np.zeros((10,784))
    for i in range(0,10):
        for j in range(0,785):
            H = np.zeros((10,784))
            H[i,j] = h
            P_plushalf = softmax(x,y,W+H/2,bias)
            P_minhalf = softmax(x,y,W-H/2,bias)
            dcdw = (NLL(P_plushalf,y) - NLL(P_minhalf,y))/h
            grad[i,j] = dcdw
            #print(grad[i,j])
            if i == 0 and j == 30:
                break
        break 
    
    grad_diff = []
    p = softmax(x,y,W,bias)
    vec_grad = gradient(p,x,y)
    for i in range(0,30):
        grad_diff.append(abs(grad[0,i] - vec_grad[0,i]))
        
    grad_diff_accum = sum(grad_diff)
    return grad_diff_accum

def get_performance_log(Y, X, W):
    '''
    Gets the performance of the data Y, X given the W matrix
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :return: percentage of correct classifications
    '''
    P = softmax(X, W)
    indices_test = np.argmax(P, 1)
    indices_actual = np.argmax(Y, 1)
    performance = (indices_test.shape[0] - np.count_nonzero(indices_test - indices_actual))/(.01*indices_test.shape[0])
    return performance

def grad_descent(Y, X, W, alpha):
    '''
    Executes gradient descent on the data set; computes the W matrix for which the cost function is minimized
    :param Y: classification matrix where each row is the one-hot encoding vector for an image (nx10 numpy array)
    :param X: flattened image matrix where each row is a flattened image (nx785 numpy array)
    :param W: weight matrix where the ith row corresponds to the weights for output i (10x785 numpy array)
    :param alpha: step size for gradient descent
    :return: W matrix for which cost is minimized
    '''
    EPS = 1e-6
    prev_W = W - 10*EPS
    max_iter = 10000
    iter = 0
    
    #X_test, Y_test = get_data('test')

    while np.linalg.norm(W-prev_W) > EPS and iter < max_iter:
        prev_W = W.copy()
        P = softmax(X,W)
        W -= alpha*gradient(P, Y, X)
        # if iter % 100 == 0:
        #     print 'Iteration:', iter
        #     print 'Cost', cost(P, Y)
        #     print 'Train Performance', get_performance(Y, X, W)
        #     print 'Test Performance', get_performance(Y_test, X_test, W)
        #     print
        iter += 1
    #c = cost(P, Y)

    #print "Minimum found at", W, "with cost function value of", c, "on iteration", iter
    return W

#Part 5:
# Coming up with a dataset where the performance on the test set is better when you use 
# multinomial logistic regression compared to Linear Regression
def df(x, y, theta):
    return -sum((y.T-dot(theta, x.T))*x.T, 1)
        
def grad_descent_lin(df, x, y, init_t, alpha):
    '''Using The Derivative of the Cost Function to Evaluate Grad. Descent To evaluate Theta.
       J(theta0,...,thetaN) - alpha*grad(J(theta0,...,thetaN))'''
       
    #Evaulating Gradient Descent 
    EPS = 1e-10 #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 10000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        #print "Iter", iter
        #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
        #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t
    
def get_performance_logp5(x,y,theta):
    a = 0
    b = 0 
    for i in range(0,len(x)):
        if i < (len(x)/2):
            if (y[i][0] - float(dot(theta,x[i].T)) ) > 0:
                a += 1 
    
        elif i > (len(x)/2):
            if (y[i][0] - float(dot(theta,x[i].T))) < 0:
                b += 1
                
    performance = (float(a+b))/(len(x))
    return performance

def get_performance_linp5(x,y,theta):
    a = 0
    b = 0 
    for i in range(0,len(x)):
        if i < (len(x)/2):
            if (y[i][0] - float(dot(theta,x[i].T)) ) > 0:
                a += 1 
    
        elif i > (len(x)/2):
            if (y[i][0] - float(dot(theta,x[i].T))) < 0:
                b += 1
                
    performance = (float(a+b))/(len(x))
    #print ("linear regression performance:", performance)
    return performance
    
def plot_line(theta, x_min, x_max, color, label):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)

theta=np.array([0.5,0.5])

#gen_data(theta,50,20,1e-5,1e-8,np.ones((1,2)))
def gen_data(theta, N, sigma,alpha1,alpha2,init_t):

    # Actual data
    x_raw = 100*(random.random((N))-.5)
    x1 = vstack((    ones_like(x_raw),
                    x_raw,
                    ))
    
    x2 = vstack((    ones_like(x_raw),
                    x_raw,
                    ))
                    
    y1 = dot(theta, x1) + scipy.stats.norm.rvs(loc = 2.5*sigma,scale=sigma,size=N)
    y2 = dot(theta, x2) - scipy.stats.norm.rvs(loc = .5*sigma,scale= sigma,size=N)
   
    # outlier 
    x_out = -50
    x_out = vstack((    ones_like(x_out),
                    x_out,
                    ))
    y_out = dot(theta,x_out) + 400

    plot(x1[1,:], y1, "bo", label = "Training set (y = 1)")
    plot(x2[1,:], y2, "ro", label = "Training set (y = 0)")
    plot(x_out[1,:],y_out,"bo")

    # Apply gradient descent and calculate performance
    # linear regression
    y_temp1 = np.ones((N+1,1))
    y_temp2 = np.zeros((N,1))
    y = np.array((2*N,1))
    y = np.vstack((y_temp1,y_temp2))
    x = np.vstack((x1.T,x_out.T))
    x = np.vstack((x,x2.T))
    t_lin = grad_descent_lin(df, x, y, init_t, alpha1)
    t_log = grad_descent(y,x,init_t,alpha2)
    
    plot_line(theta, -100, 100, "b", "Actual generating process")
    plot_line(t_lin[0], -100, 100, "g", "lin generating process")
    plot_line(t_log[0], -100, 100, "m", "log generating process")
    legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    
          fancybox=True, shadow=True, ncol=6)
    xlim([-100, 100])
    ylim([-200, 500])
    
    y1 = np.array([y1]) 
    y2 = np.array([y2])
    y3 = np.array([y_out])
    y_actual = np.vstack((y1.T,y3))
    y_actual = np.vstack((y_actual,y2.T))
    p_lin = get_performance_linp5(x,y_actual,t_lin[0])
    p_log = get_performance_logp5(x,y_actual,t_log)
    
    return plt.show(),t_lin[0],t_log[0],p_lin,p_log
    