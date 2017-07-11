import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron as P

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
df.tail()

'''
Extracted the first 100 class labels that corresponded to the 50 Iris-Setosa and 50 
Iris-Versicolor flowers, respectively, and convert the class labels into the two integer 
class labels 1(Versicolor) and -1(Setosa) that we assign to a vector y where the values 
method of a pandas DataFrame yields the corresponding NumPy representation. 
'''

y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
X=df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal-length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn=P(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()