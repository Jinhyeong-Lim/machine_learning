#import matplotlib.pyplot as plt
#plt.plot([1,2,3,4,5], [1,4,9,16,25])
#plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

height = np.array([183, 150, 180, 197, 160, 175])
height = height.reshape(-1, 1)
math = np.array([85, 45, 80, 99, 45, 75])

line_fitter = LinearRegression()
line_fitter.fit(height, math)
score_predict = line_fitter.predict(height)
plt.plot(height, math, '*')
plt.plot(height, score_predict)
plt.show()
print(line_fitter.intercept_)
print("Mean_Squared_Error :", mean_squared_error(score_predict, math))
#평균 제곱근 오차
