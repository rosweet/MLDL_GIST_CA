#heejinseo 20225085
"""
Make sure you use the method given in the question!
"""
#method given in the question is hinge loss and coordinate gradient descent.
#hinge loss: max(0, 1-y_i f(x_i)) where f(x_i) = w^T x_i + b
#coordinate gradient descent: each step updates one component (coordinate) at a time, keeping others fixed

#heejin seo 20225085
import numpy as np
from sklearn import datasets

import utils
from utils import *

iris = datasets.load_iris()
x1 = iris.data[:100, :2] #100 datas with 2 features [[0, 0],[0, 0], ..., [0, 0]]
y1 = iris.target[:100] #100 values [0, 0, ..., 1]

#work in hard-margin example
class HSVM:

    def __init__(self, alpha = 0.001, n_iterations = 40000):
        self.alpha = alpha # learning rate
        #self.lambda_ = lambda_ # tradeoff # this is for soft margin SVM!
        # number of iterations: selected empirically
        self.n_iterations = n_iterations
        self.w = None # weights or slopes
        self.b = None # intercept
    
    def fit(self, x, y):        
        n_samples, n_features = x.shape 
        weights = np.zeros(n_features) # initalizing weights with 0
        bias = 0 # initialize bias with 0
        iterations = range(self.n_iterations)
        lr = self.alpha
    
        for iter in iterations:
            #keeping other weights (for other features) fixed, 
            #update each weight for each feature. (one at a time)
            #This method is called "coordinate gradient descent"
            for w in range(len(weights)):
                #while updating the weight of each feature,
                #we'll use hinge loss.
                yixi, b = self.hinge_loss_gradient(x, y, weights, bias, w)

                #using hinge loss, update weight and bias.
                #check ppt 14, page 28, last update rule
                #"lambda" in page 28, is set to 0, since we are dealing with Hard margin SVM.
                #"lambda" will be used in Soft margin SVM. This indicates the tradeoff. 
                weights[w] -= yixi * lr
                bias -= b * lr
        #finalized weights and bias
        self.w = weights
        self.b = bias

        return weights, bias
    
    def hinge_loss_gradient(self, x, y, weights, bias, j):
        y = np.array([1 if val == 0 else -1 for val in y]) # returning in the form of -1 and 1
        yixi, b = 0, 0 #initialize to 0, since if yif(xi) >= 1, we'll just put yixi as 0.
        #for datapoints (total one hundred), we'll check whether the point is inside the margin.
        #if inside the margin, we need to project it back to constrained region, since it is a hard SVM
        for i, xi in enumerate(x):
            # max (0, 1 - yi(w^T x_i + b) ) -> hinge loss 
            if (1 - (y[i] * (np.dot(xi, weights) + bias))) > 0:
                # ppt 14, page 28, last update rule, of pegasos algorithm.
                yixi -= y[i] * xi[j]
                b -= y[i]
        return yixi, b

    def predict(self, x):
        # our final output
        pred = np.dot(x, self.w) + self.b 
        # returning in the form of 0 and 1
        return np.array([0 if val > 0 else 1 for val in pred])

