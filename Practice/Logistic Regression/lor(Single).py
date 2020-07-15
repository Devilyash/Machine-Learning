# for classification problem

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('/home/devilyash/Documents/Machine Learning/Practice/Logistic Regression/hr_analytics.csv')

left = df[df.left==1]
retained = df[df.left==0]

df.groupby('left').mean()

pd.crosstab(df.salary, df.left).plot(kind='bar')
#plt.show()
pd.crosstab(df.Department, df.left).plot(kind='bar')

# dfle = df
# le = LabelEncoder()
# dfle.salary = le.fit_transform(dfle.salary)
# X = dfle[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']].values
# ct = ColumnTransformer([('salary', OneHotEncoder(), [0])], remainder='passthrough')
# X = ct.fit_transform(X)
# X = X[:,1:]
# y = dfle.left.values

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummies = pd.get_dummies(subdf.salary, prefix='salary')
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
X = df_with_dummies
y = df.left

x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

reg = LogisticRegression()
reg.fit(x_train, y_train)
print(reg.predict(x_test))
print(reg.score(x_test, y_test))



