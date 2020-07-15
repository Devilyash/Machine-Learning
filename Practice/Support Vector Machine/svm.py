# it is a classification algorithm

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

# print(iris.feature_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())
df['target'] = iris.target
# iris.target_names
# df[df.target==1].head()  #0 = setosa, 1 = versicolor, 2 = virginica

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

#lets do data visualization
from matplotlib import pyplot as plt
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

# plt.xlabel('sepal length (cm)')
# plt.ylabel('sepal width (cm)')
# plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green',marker='+')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='red',marker='.')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green',marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='red',marker='.')
# plt.show()

from sklearn.model_selection import train_test_split