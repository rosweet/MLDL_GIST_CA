#heejinseo 20225085
import numpy as np
from sklearn import datasets

from numpy import random

import utils
from utils import *

iris=datasets.load_iris()
x2=iris.data[50:,[2,3]] #50 datas with 2 features
y2=iris.target[50:] #50 values
#[1 1 1... 1 2 2 2 ... 2]

#################################################
#class for Primal problem: SSVM
class SSVM:
    def __init__(self, alpha = 0.001, lmda = 0.01, n_iterations = 1000):
        self.alpha = alpha #learning rate

        self.lmba = lmda #tradeoff
        #If lmda is too low, this model becomes hard margin
        # since lmda is a "tradeoff"
        # between the margin size and datapoint being on the correct side based on the boundary.
        
        #Thus, lmda does the role of "slack variable" which allows misclassification.
        #If lmda is small, we don't allow datapoints to be inside the margin. 
        #=> Thus, margin size gets smaller, but all datapoints are on the correct side.
        #If lmda is big, we allow datapoints to be inside the margin. 
        #=>Thus, margin size gets bigger, allowing mistakes.

        #Also, changing lmda changes the decision boundary, so the best lmda can be found empirically!
        #-> for example, lmba = 0.01 and lmba = 0.02 yields different decision boundaries.
        
        self.n_iterations = n_iterations
        #f(x) = w^T x + b
        self.w = None # weights or slopes
        self.b = None #intercept
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features) #initializing weights with 0
        self.b = 0 #initialize bias with 0
        iterations = range(self.n_iterations)
        lr = self.alpha
        trdoff = self.lmba
        
        for iter in iterations:
            #keeping other weights (for other features) fixed, 
            #update each weight for each feature. (one at a time)
            #This method is called "coordinate gradient descent"
            for n in range(len(self.w)):
                #while updating the weight of each feature,
                #we'll use hinge loss.
                yixi, b = self.hinge_loss_gradient(x, y, self.w, self.b, n)

                #using hinge loss, update weight and bias.
                #check ppt 14, page 28, last update rule
                #"lambda" in page 28, is set to 0.01, since we are dealing with soft margin SVM.
                # THIS LAMBDA(trdoff) is the ONLY difference between my HSVM and SSVM.
                self.w[n] -= lr*(yixi - (trdoff*self.w[n]))
                self.b -= b * lr

        return self.w, self.b
    
    def hinge_loss_gradient(self, x, y, weights, bias, j):
        #y = np.array([1 if val == 0 else -1 for val in y]) # returning in the form of -1 and 1
        y = np.array([-1 if val == 1 else 1 for val in y]) # returning in the form of -1 and 1
        yixi, b = 0, 0 #initialize to 0, since if yif(xi) >= 1, we'll just put yixi as 0.
        #for datapoints (total one hundred), we'll check whether the point is inside the margin.
        #if inside the margin, change dL/dw and the bias term
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
        # returning in the form of 2 and 1
        return np.array([2 if val > 0 else 1 for val in pred])

#################################################
#class for Dual problem: SVM_dual
class SVM_dual: 

    def __init__(self, n_iterations=1000, lr= 0.05):

        self.alpha = None # the lagrange multiplier # the "dual" alpha.
        self.C = 1 # constraint of alpha # 0 < alpha < C

        self.b = 0 # initialize bias(intercept) to 0

        self.n_iterations = n_iterations

        self.lr = lr #learning rate
      
    def linear(self, X, Z):
        #linear kernels = just the inner product ("x_i * x_j" in the original dual problem)
        #K(u, v) = X dot Z
        return X.dot(Z.T)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        self.alpha = np.random.random(n_samples)

        self.X = X
        self.y = y
        self.b = 0 #initialize bias to 0
                       
        # our wolfe dual problem (L_D): ∑ alpha_i - (1/2)∑ alpha_i alpha_j y_i y_j x_i*x_j
        # K(x_i, x_j) = x_i*x_j
        # L_D = ∑ alpha_i - (1/2)∑ alpha_i alpha_j y_i y_j K(x_i, x_j)

        # differentiating L_D in k^th term of alpha = gradient
        # 1 - y_k ∑ alpha_j y_j x_i*x_j = 1 - y_k ∑ alpha_j y_j K(x_i, x_j)

        #update alpha: coordinate gradient descent
        #K(xi, xj): linear kernel function,
        #xi innerproduct xj: self.linear(X, X)
        for iter in range(self.n_iterations):
            ykyj = np.outer(y, y)
            K_xixj = self.linear(X, X)
            non_alpha_term = ykyj * K_xixj

            L_D_dirrerentiate = [1]*n_samples - non_alpha_term.dot(self.alpha) # 1 – y_k ∑ αj yj K(x_j, x_k)

            # alpha = alpha + learning_rate*(1 – y_k ∑ alpha_j y_j K(x_j, x_k)) is the update rule that GD iterates
            self.alpha += self.lr * L_D_dirrerentiate 

            # 0 <= alpha <= C
            # projected gradient descent: if gradient update is leaving the constrained area, project it back
            for i in range(len(self.alpha)):
                if self.alpha[i] < 0: #gradient update is leaving the lower bound of constrained area
                    self.alpha[i] = 0 #project it back
            for i in range(len(self.alpha)):
                if self.alpha[i] > self.C: #gradient update is leaving the upper bound of constrained area
                    self.alpha[i] = self.C #project it back

            #our wolfe dual problem, but not used...
            aiaj = np.outer(self.alpha, self.alpha)    
            l_d = np.sum(self.alpha) - (1/2)* np.sum(aiaj * ykyj * K_xixj) # ∑alpha_i – (1/2) ∑ij alpha_i alpha_j y_i y_j K(x_i, x_j)

        constraint_alpha_idx = [] #index of alphas that are inside the constrained area
        for idx, value in enumerate(self.alpha):
            if 0 < value:
                if value < self.C:
                    constraint_alpha_idx.append(idx)
        constraint_alpha_idx = np.array(constraint_alpha_idx)

        # computing bias #bias = average (yi – ∑ α_j _yj K(x_j, x_i)) for alphas inside the constrained area
        biases = []        
        for idx in constraint_alpha_idx:
            k_xjxi = self.linear(X, X[idx])
            biases.append(y[idx] - (self.alpha * y).dot(k_xjxi))

        self.b = np.sum(biases) / len(biases)

    # dual classifier (ppt 14, 20 page): f(x) = ∑ alpha_i y_i (X_i^T X) + b
    def decision_boundary(self, X):
        xtx = self.linear(self.X, X)
        dual_classifier = (self.alpha*self.y).dot(xtx) + self.b
        return dual_classifier
                
    def predict(self, X):
        pred_y = np.array([2 if val > 0 else 1 for val in self.decision_boundary(X)])
        return pred_y
