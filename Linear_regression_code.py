import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
data=pd.read_csv('Customer Purchase Data.csv')
print(data.describe())
print(data.info())
dummies=pd.get_dummies(data,columns=['Age'],dtype='int64')
print(dummies)
data1=dummies.drop(columns=['Number'])
print(data1)
print(data1.columns)
print(data1.corr().to_string())
X=data1[["Income", "Membership_Years"]]
Y=data1['Purchase_Frequency']
print(X)
print(Y)
sns.scatterplot(data=data1,x='Income',y='Purchase_Frequency')
plt.plot(X,Y)
plt.show()
random.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
regr= LinearRegression()
regr.fit(X_train, Y_train)
print(regr.score(X_test, Y_test))