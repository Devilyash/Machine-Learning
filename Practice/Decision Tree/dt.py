import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('~/Documents/Machine Learning/Practice/Decision Tree/salaries.csv')

inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company','job','degree'],axis='columns')

x_train, x_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)
# print(model.score(x_train, y_train))
model.predict([[2,0,1]])

y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

# from sklearn.tree import export_graphviz
# export_graphviz(model, out_file ='tree.dot', 
#                feature_names =['company_n', 'job_n', 'degree_n'])
