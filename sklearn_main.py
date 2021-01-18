import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
# Regression Lineaire avec SKLearn
np.random.seed(0)
m = 100
X = np.linspace(0, 10, m).reshape(m, 1)
y = X**2 + np.random.randn(m, 1)  

model = SVR(C=100)
# model = LinearRegression()
model.fit(X, y)
model.score(X, y)
predict = model.predict(X)


plt.scatter(X, y)
plt.plot(X, predict, c='g')
plt.show()