import numpy as np

# you can modify util functions here what you need
# this python file will not be included in the grading

import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt

def computeClassificationAcc(y_true, y_pred):
  #check whether the lengths are same
  if len(y_true) != len(y_pred):
    raise ValueError("y_true and y_pred must have same length.")

  #COUNT correctly classified samples
  correct = np.sum(y_true == y_pred)

  #Total number of samples
  total = len(y_true)

  #Fraction of accurate samples
  accuracy = correct / total

  return accuracy 

#usage: plot the decision boundary of soft margin SVM
def plot_decision_boundary_hsvm(model, title):

  iris = datasets.load_iris()
  X = iris.data[:100, :2]
  y = iris.target[:100]
  plt.figure(figsize=(10, 5))
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, s=50, edgecolors='k')
  
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  
  # Create grid to evaluate model
  xx = np.linspace(xlim[0], xlim[1], 300)
  yy = np.linspace(ylim[0], ylim[1], 300)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  #Z = model.predict(xy).reshape(XX.shape)
  arr = np.array(model.predict(xy)) 
  Z = arr.reshape(XX.shape) 
  
  # Plot decision boundary and margins
  ax.contour(XX, YY, Z, colors='k', levels=[0, 1, 2], alpha=0.5)
  plt.title(title)
  plt.xlabel('Length (cm)')
  plt.ylabel('width (cm)')
  plt.show()  
  return None

#usage: plot the decision boundary of soft margin SVM
def plot_decision_boundary_ssvm(model, title):

  iris = datasets.load_iris()
  X=iris.data[50:,[2,3]]
  y=iris.target[50:]
  plt.figure(figsize=(10, 5))


  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=.5)
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  xx = np.linspace(xlim[0], xlim[1], 300)
  yy = np.linspace(ylim[0], ylim[1], 300)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  arr = np.array(model.predict(xy)) 
  Z = arr.reshape(XX.shape)

  ax.contour(XX, YY, Z, levels=[0, 1, 2],linestyles=['--', '-', '--'])
  plt.title(title)
  plt.show()  

  return None

def plot_decision_boundary_kernel_svm(model, title):

  dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20)
  X = dataset[0]
  y = dataset[1]

  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=.5)
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  xx = np.linspace(xlim[0], xlim[1], 300)
  yy = np.linspace(ylim[0], ylim[1], 300)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  arr = np.array(model.predict(xy)) 
  Z = arr.reshape(XX.shape) 

  #we must indicate support vectors, which are the datapoints lying within the margin(=1).
  decision_function = model.decision_boundary(X)
  support_vector_index = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
  support_vectors = X[support_vector_index]

  ax.contour(XX, YY, Z, levels=[0, 1, 2],linestyles=['--', '-', '--'])

  plt.scatter(
      support_vectors[:, 0],
      support_vectors[:, 1],
      s=100,
      linewidth=1,
      facecolors="none",
      edgecolors="r",
    )
  
  plt.title(title)
  plt.show()  

  return None

def plot_decision_boundary_kernel_svm_sklearn(model, title):

  dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20)
  X = dataset[0]
  y = dataset[1]

  plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=.5)
  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  xx = np.linspace(xlim[0], xlim[1], 300)
  yy = np.linspace(ylim[0], ylim[1], 300)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  arr = np.array(model.predict(xy)) 
  Z = arr.reshape(XX.shape) 

  #we must indicate support vectors, which are the datapoints lying within the margin(=1).
  decision_function = model.decision_function(X)
  support_vector_index = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
  support_vectors = X[support_vector_index]

  ax.contour(XX, YY, Z, levels=[0, 1, 2],linestyles=['--', '-', '--'])

  plt.scatter(
      support_vectors[:, 0],
      support_vectors[:, 1],
      s=100,
      linewidth=1,
      facecolors="none",
      edgecolors="r",
    )
  
  plt.title(title)
  plt.show()  

  return None