import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def modele(X, theta):
    return X.dot(theta)

def cout(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((modele(X,theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(modele(X,theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cout(X, y, theta)
    return theta, cost_history

def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()

    return 1 - u/v

def show_3d(x, y, pred):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x[:,0], x[:,1], y)
    ax.scatter(x[:,0], x[:,1], pred)
    plt.show()


x, y = make_regression(n_samples=100, n_features=2, noise=10)
# y = y - abs(y/2)
y = y.reshape(y.shape[0], 1)
X = np.hstack((x, np.ones((x.shape[0], 1))))
theta = np.random.randn(3, 1)

n = 1000
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=n)

prediction = modele(X, theta_final)

print(coef_determination(y, prediction))

# plt.scatter(x[:,1], y, c='r')
# plt.scatter(x[:,1], prediction, c='g')
# plt.show()

show_3d(x, y, prediction)
