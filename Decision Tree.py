import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #testset,train set 나눔
from sklearn.tree import DecisionTreeClassifier #의사 결정 나무를 불러옴
from sklearn import metrics
from sklearn. tree import export_graphviz
import pydotplus
from IPython.display import Image
import os
os.environ["PATH"]+=os.pathsep + 'c:/Program Files (x86)/Graphviz2.38/bin/'


iris = load_iris()
x=iris.data
y=iris.target
print(y)
df = pd.DataFrame(x, columns=['sepal_width(cm)', 'sepal_length(cm)', 'petal_width(cm)', 'petal_length(cm)'])
#print(df)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
#data 중에서 70%는 training data고 30%가 test data

clf = DecisionTreeClassifier() #decision tree 생성
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test) #test data set을 통해 예측

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

