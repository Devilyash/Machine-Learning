#For Both regression and Classification as Decision Tree
#For making many Decision tree

import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40) 
model.fit(x_train, y_train)

# print(model.score(x_test, y_test)) #96.67%

y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
# print(cm)

import matplotlib.pyplot as plt
import seaborn as sm
plt.figure(figsize=(10,7))
sm.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()