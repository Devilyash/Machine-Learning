import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('/home/devilyash/Documents/Machine Learning/Practice/Linear Regression/homeprices.csv')

# %matplotlib inline
plt.xlabel('area')
plt.ylabel('price($US')
plt.scatter(df.area, df.price, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
print(reg.predict([[3300]]))

print(reg.coef_)
print(reg.intercept_)

d = pd.read_csv('/home/devilyash/Documents/Machine Learning/Practice/Linear Regression/area.csv')
print(d)
p = reg.predict(d)
print(p)
d['prices'] = p

d.to_csv('/home/devilyash/Documents/Machine Learning/Practice/Linear Regression/area.csv')
