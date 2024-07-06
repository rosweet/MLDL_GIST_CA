#########################################################   
###############< test for SVM_hard >#####################
import numpy as np
from sklearn import datasets

import utils
from utils import *

iris = datasets.load_iris()
x1 = iris.data[:100, :2] #100 datas with 2 features [[0, 0],[0, 0], ..., [0, 0]]
y1 = iris.target[:100] #100 values [0, 0, ..., 1]

from SVM_hard import *
from utils import *

#########################  

model1 = HSVM()
model1.fit(x1, y1)
y_pred = model1.predict(x1)
acc = computeClassificationAcc(y_pred, y1) 
print("Accuracy of HSVM:", acc)
utils.plot_decision_boundary_hsvm(model1, "hard margin SVM")
#accuracy: 1.0


#########################################################   
###############< test for SVM_soft >#####################
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[50:, [2,3]]
y = iris.target[50:] 

from sklearn.model_selection import train_test_split
tr_x, val_x, tr_y, val_y = train_test_split(X, y, test_size = 0.3, random_state = 40)

from SVM_soft import *
from utils import *

#########################  

model2 = SSVM()
model2.fit(tr_x, tr_y)
y_pred = model2.predict(val_x)
acc = computeClassificationAcc(val_y, y_pred) 
print("Accuracy of Primal SSVM:", acc)
utils.plot_decision_boundary_ssvm(model2, "primal soft margin SVM")
#accuracy: 0.9

#########################  
#Dual problem of Soft Margin SVM

model3 = SVM_dual()
model3.fit(tr_x, tr_y)
y_pred = model3.predict(val_x)
acc = computeClassificationAcc(val_y, y_pred) 
print("Accuracy of Dual SSVM:", acc)
utils.plot_decision_boundary_ssvm(model3, "dual soft margin SVM")
#accuracy: 0.9

###########################################################   
###############< test for SVM_kernel >#####################
import sklearn
from sklearn import datasets

dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
x = dataset[0]
y = dataset[1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

from SVM_kernel import *
from utils import *

#########################  
svm3 = KSVM(kernel_type = "rbf")
svm3.fit(X_train,y_train)
print("Accuracy of SVM using rbf:", utils.computeClassificationAcc(svm3.predict(X_test), y_test))
utils.plot_decision_boundary_kernel_svm(svm3, "kernel SVM, rbf")
# accuracy: 0.866...67

svm4 = KSVM(kernel_type = "polynomial_upto_d", degree = 3, poly_c = -2)
svm4.fit(X_train,y_train)
print("Accuracy of SVM using polynomial with degree upto d: ", utils.computeClassificationAcc(svm4.predict(X_test), y_test))
utils.plot_decision_boundary_kernel_svm(svm4, "kernel SVM, polynomial upto d")
# accuracy:
#degree = 2, c = -0.3 : ?
#degree = 1, c = -0.3 : ?
#degree = 3, c = -2 : 0.73...3
# we can get better kernel empirically

svm5 = KSVM(kernel_type = "linear")
svm5.fit(X_train,y_train)
print("Accuracy of SVM using linear kernel: ", utils.computeClassificationAcc(svm5.predict(X_test), y_test))
utils.plot_decision_boundary_kernel_svm(svm5, "kernel SVM, linear")
# accuracy: 0.722..2

svm6 = KSVM(kernel_type = "polynomial_exactly_d", degree = 2)
svm6.fit(X_train,y_train)
print("Accuracy of SVM using polynomial with degree exactly d: ", utils.computeClassificationAcc(svm6.predict(X_test), y_test))
utils.plot_decision_boundary_kernel_svm(svm6, "kernel SVM, polynomial exactly d")
# accuracy: 0.577...7 when degree = 2

svm7 = KSVM(kernel_type = "sigmoid", sig_a=5, sig_b=-1)
svm7.fit(X_train,y_train)
print("Accuracy of SVM using sigmoid kernel: ", utils.computeClassificationAcc(svm7.predict(X_test), y_test))
utils.plot_decision_boundary_kernel_svm(svm7, "kernel SVM, sigmoid")
# accuracy: 0.7 when sig_a = 5, sig_b = -1
