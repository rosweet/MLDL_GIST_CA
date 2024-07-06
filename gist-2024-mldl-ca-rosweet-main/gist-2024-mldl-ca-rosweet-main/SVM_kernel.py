#heejinseo 20225085
import numpy as np

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

from numpy import random

import utils
from utils import *

dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
x = dataset[0]
y = dataset[1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
#y: [0 0 0... 0 1 1 1 ... 1]

class KSVM: 

    def __init__(self, sigma=0.1, n_iterations=1000, lr= 0.001, kernel_type = "rbf", degree = 0, poly_c = 0, sig_a = 0, sig_b = 0):

        self.alpha = None # the lagrange multiplier # the "dual" alpha.
        self.C = 1 # constraint of alpha # 0 < alpha < C

        self.b = 0 # initialize bias(intercept) to 0

        self.sigma = sigma # hyperparameter for rbf

        self.poly_c = poly_c # hyperparameter for polynomial-kernels
        self.degree = degree # hyperparameter for polynomial-kernels

        self.sig_a = sig_a #hyperparameter for sigmoid-kernels
        self.sig_b = sig_b #hyperparameter for sigmoid-kernels

        self.n_iterations = n_iterations

        self.lr = lr #learning rate

        if kernel_type == "rbf":
            self.kernel_type = self.rbf # we are going to use gaussian kernel.
        elif kernel_type == "polynomial_upto_d":
            self.kernel_type = self.polynomial_upto_d
        elif kernel_type == "linear":
            self.kernel_type = self.linear
        elif kernel_type == "polynomial_exactly_d":
            self.kernel_type = self.polynomial_exactly_d
        elif kernel_type == "sigmoid":
            self.kernel_type = self.sigmoid
    
        
    def rbf(self, X, Z):
        #gaussain kernels
        #K(u, v) = exp(-||X-y||^2_2 / (2 sigma^2)) #uses euclidean norm
        l2_norm = np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2)
        l2_norm_squared = l2_norm ** 2
        kernel_bandwidth = 1 / self.sigma ** 2 #hyperparameter, ppt 15 page 20
        k_uv = np.exp(-l2_norm_squared * kernel_bandwidth)
        return k_uv
    
    def polynomial_upto_d(self, X, Z):
        #polynomial kernels
        #K(u, v) = (X dot Z + c)**self.degree
        return (X.dot(Z.T) + self.poly_c)**self.degree
    
    def polynomial_exactly_d(self, X, Z):
        #polynomial kernels
        #K(u, v) = (X dot Z)**self.degree
        return (X.dot(Z.T))**self.degree
    
    def linear(self, X, Z):
        #linear kernels
        #K(u, v) = X dot Z
        return X.dot(Z.T)
    
    def sigmoid(self, X, Z):
        #linear kernels
        #K(u, v) = tanh (sig_a (X dot Z) + sig_b)
        return np.tanh(self.sig_a * X.dot(Z.T)) + self.sig_b
    
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
        #K(xi, xj): gausian kernel function,
        #xi innerproduct xj: self.kernel_type(X, X)
        for iter in range(self.n_iterations):
            ykyj = np.outer(y, y)
            K_xixj = self.kernel_type(X, X)
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
            if 0 <= value:
                if value <= self.C:
                    constraint_alpha_idx.append(idx)
        constraint_alpha_idx = np.array(constraint_alpha_idx)

        # computing bias #bias = average (yi – ∑ α_j _yj K(x_j, x_i)) for alphas inside the constrained area
        biases = []        
        for idx in constraint_alpha_idx:
            k_xjxi = self.kernel_type(X, X[idx])
            biases.append(y[idx] - (self.alpha * y).dot(k_xjxi))

        self.b = np.sum(biases) / len(biases)

    # dual classifier (ppt 14, 20 page): f(x) = ∑ alpha_i y_i (X_i^T X) + b
    def decision_boundary(self, X):
        xtx = self.kernel_type(self.X, X)
        dual_classifier = (self.alpha*self.y).dot(xtx) + self.b
        return dual_classifier
                
    def predict(self, X):
        # returning in the form of 0 and 1
        pred_y = np.array([0 if val <= 0 else 1 for val in self.decision_boundary(X)])
        return pred_y


"""
#code for comparing the accuracy of common kernels, using "training set" 
#-> this cannot indicate the actual accuracy of model (using test set), but it can be used to grasp the efficiency of models roughly.
    
svm3 = KSVM(kernel_type = "rbf")
svm3.fit(x,y)
print("Accuracy: ", utils.computeClassificationAcc(svm3.predict(x), y))
utils.plot_decision_boundary_kernel_svm(svm3, "kernel SVM, rbf")
# Accuracy: 0.88

svm4 = KSVM(kernel_type = "polynomial_upto_d", degree = 3, poly_c = -2)
svm4.fit(x,y)
print("Accuracy: ", utils.computeClassificationAcc(svm4.predict(x), y))
utils.plot_decision_boundary_kernel_svm(svm4, "kernel SVM, polynomial upto d")
# Accuracy:
# degree = 2, c = -0.3 : 0.32666...
# degree = 1, c = -0.3 : 0.73
# degree = 3, c = -2 : 0.7066667...
# we can get better kernel empirically

svm5 = KSVM(kernel_type = "linear")
svm5.fit(x,y)
print("Accuracy: ", utils.computeClassificationAcc(svm5.predict(x), y))
utils.plot_decision_boundary_kernel_svm(svm5, "kernel SVM, linear")
# Accuracy: 0.7

svm6 = KSVM(kernel_type = "polynomial_exactly_d", degree = 2)
svm6.fit(x,y)
print("Accuracy: ", utils.computeClassificationAcc(svm6.predict(x), y))
utils.plot_decision_boundary_kernel_svm(svm6, "kernel SVM, polynomial exactly d")
# Accuracy: 0.54666... when degree = 2

svm7 = KSVM(kernel_type = "sigmoid", sig_a=5, sig_b=-1)
svm7.fit(x,y)
print("Accuracy: ", utils.computeClassificationAcc(svm7.predict(x), y))
utils.plot_decision_boundary_kernel_svm(svm7, "kernel SVM, sigmoid")
# Accuracy: 0.69666...67 when sig_a = 5, sig_b = -1

"""