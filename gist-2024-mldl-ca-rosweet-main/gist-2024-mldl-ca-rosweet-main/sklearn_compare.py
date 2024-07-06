#python code for discussion part
#using the same parameter with "SVM_kernel"

import numpy as np

import sklearn
from sklearn import datasets

#cannot use this code
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import utils

###############################
##### comparing with HSVM #####

iris = datasets.load_iris()
X = iris.data[:100, :2] #100 datas with 2 features [[0, 0],[0, 0], ..., [0, 0]]
y = iris.target[:100] #100 values [0, 0, ..., 1]

# split into test dataset and train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#hard margin svm: regularization parameter C very large = 10^10
hard_svm_model = SVC(kernel="linear", C = float(10**10))
hard_svm_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_hard = hard_svm_model.predict(X_test)

accuracy_hard = accuracy_score(y_test,y_pred_hard)
print("hardSVM accuracy using sklearn:",accuracy_hard)

utils.plot_decision_boundary_hsvm(hard_svm_model, 'Hard Margin SVM using sklearn')

###############################
##### comparing with SSVM #####

iris = datasets.load_iris()
X = iris.data[50:,[2,3]]
y = iris.target[50:] #100 values [0, 0, ..., 1]

# split into test dataset and train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#soft margin svm: regularization parameter C small = 1
soft_svm_model = SVC(kernel="linear", C = 1.0)
soft_svm_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_soft = soft_svm_model.predict(X_test)

accuracy_soft = accuracy_score(y_test,y_pred_soft)
print("softSVM accuracy using sklearn:",accuracy_soft)

utils.plot_decision_boundary_ssvm(soft_svm_model, 'Soft Margin SVM using sklearn')

###################################
##### comparing with SVM_dual #####

iris = datasets.load_iris()
X = iris.data[50:,[2,3]]
y = iris.target[50:] #100 values [0, 0, ..., 1]

# split into test dataset and train dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dual_svm_model = SVC(kernel="linear", C = 1.0)
dual_svm_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_dual = dual_svm_model.predict(X_test)

accuracy_dual = accuracy_score(y_test,y_pred_dual)
print("softSVM accuracy using sklearn:",accuracy_dual)

utils.plot_decision_boundary_ssvm(dual_svm_model, 'dual SVM using sklearn')

#####################################
##### comparing with SVM_kernel #####

dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
x = dataset[0]
y = dataset[1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

#rbf
rbf_model = SVC(kernel="rbf", C = 1.0, gamma = 0.1)
#gamma is the same as sigma in "SVM_kernel.py"
rbf_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_rbf = rbf_model.predict(X_test)

accuracy_rbf = accuracy_score(y_test,y_pred_rbf)
print("SVM using RBF kernel's accuracy, using sklearn:",accuracy_rbf)

utils.plot_decision_boundary_kernel_svm_sklearn(rbf_model, 'SVM using RBF kernel')

#linear
lin_model = SVC(kernel="linear", C = 1.0)
lin_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_lin = lin_model.predict(X_test)

accuracy_lin = accuracy_score(y_test,y_pred_lin)
print("SVM using linear kernel's accuracy, using sklearn:",accuracy_lin)

utils.plot_decision_boundary_kernel_svm_sklearn(lin_model, 'SVM using linear kernel')

#polynomial exactly d (d=2)
poly_model = SVC(kernel="poly", C = 1.0, degree = 2)
poly_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_poly = poly_model.predict(X_test)

accuracy_poly = accuracy_score(y_test,y_pred_poly)
print("SVM using polynomial kernel's accuracy, using sklearn:",accuracy_poly)

utils.plot_decision_boundary_kernel_svm_sklearn(poly_model, 'SVM using polynomial kernel')

#polynomial upto d (d=3, coef0 = -2)
poly_model = SVC(kernel="poly", C = 1.0, degree = 3, coef0 = -2)
poly_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_poly = poly_model.predict(X_test)

accuracy_poly = accuracy_score(y_test,y_pred_poly)
print("SVM using polynomial kernel's accuracy, using sklearn:",accuracy_poly)

utils.plot_decision_boundary_kernel_svm_sklearn(poly_model, 'SVM using polynomial kernel')

#sigmoid
sigmoid_model = SVC(kernel="sigmoid", C = 1.0, gamma = 5, coef0 = -1)
sigmoid_model.fit(X_train,y_train)
#using the function predict(), Test! 
y_pred_sigmoid = sigmoid_model.predict(X_test)

accuracy_sigmoid = accuracy_score(y_test,y_pred_sigmoid)
print("SVM using sigmoid kernel's accuracy, using sklearn:",accuracy_sigmoid)

utils.plot_decision_boundary_kernel_svm_sklearn(sigmoid_model, 'SVM using sigmoid kernel')