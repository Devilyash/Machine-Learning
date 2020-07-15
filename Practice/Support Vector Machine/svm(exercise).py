import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)
df = pd.DataFrame(digits.data, digits.target)
df['target'] = digits.target
# print(df.head())'

x = df.drop('target', axis='columns')
y = df.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')
rbf_model.fit(x_train, y_train)
print(rbf_model.score(x_test, y_test))

linear_model = SVC(kernel='linear')
linear_model.fit(x_train,y_train)
print(linear_model.score(x_test, y_test))