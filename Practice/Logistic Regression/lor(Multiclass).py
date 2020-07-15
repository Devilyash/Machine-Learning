import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Exercise
# digits = load_iris()
# dir(digits)
# # print(digits.target[1])

# x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target ,test_size=0.2, random_state=100)
# model = LogisticRegression(max_iter=10000)
# model.fit(x_train, y_train)
# print(model.score(x_test,y_test))


#Digits dataset
digits = load_digits()

dir(digits)
digits.data[0]

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
# plt.show()


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
 

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)
# print(model.score(x_test, y_test))

# plt.matshow(digits.images[16])
# print(digits.target[16]) #6

model.predict([digits.data[16]])

y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

