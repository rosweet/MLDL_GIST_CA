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
class SVM_extra:
    def __init__(self, alpha = 0.001, lmda = 0.01, n_iterations = 500):
        self.alpha = alpha #learning rate

        self.lmba = lmda #tradeoff
        # If lmda is too low, this model becomes hard margin. 
        # since lmda is a tradeoff between the margin size and datapoint being on the correct side of margin.

        # If we want to implement Hard Margin SVM, we should put lmda as 0.
        # By default, it is given as 0.01

        # Changing lmda changes the decision boundary, so the best lmda can be found empirically!
        #-> for example, lmba = 0.01 and lmba = 0.02 yields different decision boundaries.
        
        self.n_iterations = n_iterations
        #f(x) = w^T x + b
        self.w = None # weights or slopes
        self.b = None #intercept
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        weights = np.zeros(n_features) #initializing weights with 0
        bias = 0 #initialize bias with 0
        iterations = range(self.n_iterations)
        lr = self.alpha
        trdoff = self.lmba
        
        for iter in iterations:
            #keeping other weights (for other features) fixed, 
            #update each weight for each feature. (one at a time)
            #This method is called "coordinate gradient descent"
            for w in range(len(weights)):
                #while updating the weight of each feature,
                #we'll use hinge loss.
                yixi, b = self.no_hinge_loss_gradient(x, y, weights, bias, w)

                #using hinge loss, update weight and bias.
                #check ppt 14, page 28, last update rule
                #"lambda" in page 28, is set to 0.01, since we are dealing with soft margin SVM.
                # THIS LAMBDA(trdoff) is the ONLY difference between my HSVM and SSVM.
                weights[w] -= lr*(yixi - (trdoff*weights[w]))
                bias -= b * lr

        #finalized weights and bias
        self.w = weights
        self.b = bias

        return weights, bias
    
    # < Challenge for extra credit >

    # PPT 13 page 22 introduces "hinge loss" function for optimization method.
    # However, it is written that "it doesn't care how confiently the classification is correct".
    # My idea of this extra.py file was to give reward for the correctly classified datapoint.

    # The name of the function changed from "hinge_loss_gradient" to "no_hinge_loss_gradient" since it doesn't have "hinge" shape no more. :)
    # Originally, hinge loss function was max(0, 1 - y_i f(x_i)) which looked like a slightly closed hinge,
    # but now it is a fully opened hinge, which looks like -y_i f(x_i) !

    def no_hinge_loss_gradient(self, x, y, weights, bias, j):
        y = np.array([-1 if val == 1 else 1 for val in y]) # returning in the form of -1 and 1
        yixi, b = 0, 0 #initialize to 0.
        #for datapoints (total one hundred), we'll check whether the point is inside the margin.
        #if inside the margin, give penalty as before.
        #if correct (outside the margin), give reward (in the opposite direction of penalty)
        for i, xi in enumerate(x):
            # max (0, 1 - yi(w^T x_i + b) ) -> hinge loss 
            if (1 - (y[i] * (np.dot(xi, weights) + bias))) > 0:
                # ppt 14, page 28, last update rule, of pegasos algorithm.
                yixi -= y[i] * xi[j]
                b -= y[i]
            else: 
                #if 1 - yi(w^T x_i + b) < 0 
                # ==> 1 < yi(w^T x_i + b)
                # does care "how confidently the classification is correct".
                yixi += y[i] * xi[j]
                b += y[i]                
        return yixi, b

    def predict(self, x):
        # our final output
        pred = np.dot(x, self.w) + self.b
        # returning in the form of 2 and 1
        return np.array([2 if val > 0 else 1 for val in pred])

#################################################
#the test code for extra.py

from sklearn.model_selection import train_test_split
tr_x, val_x, tr_y, val_y = train_test_split(x2, y2, test_size = 0.3, random_state = 40)

svm_ext = SVM_extra()

svm_ext.fit(tr_x, tr_y)
utils.plot_decision_boundary_ssvm(svm_ext, "extra model")

accuracy = utils.computeClassificationAcc(svm_ext.predict(val_x), val_y) 
print("The accuracy of the extra model is", accuracy)
#accuracy: 0.966...67
#working well!


#We can see that this new model is even working well than the original SVM model.
#Using the same n_iterations (epoches = 500), SSVM obtained 0.9333...3 accuracy, but our extra model obtained 0.966..67 accuracy.
#The original SVM focused only on putting every datapoint "at least" outside the margin.
#That is, we wanted our model to put all points in a way that y*(wx+b) is at least 1.
#We haven't cared how "confidently" or how "correctly" the correct datapoints were classified.
#Now, considering how "confidently" the datapoints were classified, we can reduce the possibility of the model being overfitted.
#Thus, we have higher test accuracy.
