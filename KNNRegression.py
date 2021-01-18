from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas

dataset = pandas.read_csv('https://www-lisic.univ-littoral.fr/~teytaud/files/Cours/Apprentissage/data/auto-mpg.data', sep=',')
dataset = dataset[['mpg','cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','year','origin']]
dataset.dropna(axis=0, inplace=True)
X = dataset.drop(columns=['mpg'])
y = dataset['mpg']

model = KNeighborsRegressor(n_neighbors=7)
model.fit(X, y)
print(model.score(X, y))
predict = model.predict(X)

plt.scatter(dataset['cylinders'], predict, c='r')
plt.scatter(dataset['horsepower'], predict, c='g')
plt.show()
#scores = cross_val_score(model, X, y ,cv=5, scoring="accuracy")

#print(scores)
#print(scores.mean())