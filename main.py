import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes();

diabetes_x = diabetes.data
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]
diabetes_x_test_plot = diabetes_x[-30:, np.newaxis, 2]
diabetes_y_test: object = diabetes.target[-30:]
diabetes_y_train = diabetes.target[:-30]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

diabetes_Y_predict = model.predict(diabetes_x_test)

print("mean square error: ", mean_squared_error(diabetes_y_test, diabetes_Y_predict))

# plt.scatter(diabetes_x_train, diabetes_y_train, color='red')
plt.plot(diabetes_Y_predict)

plt.show()
