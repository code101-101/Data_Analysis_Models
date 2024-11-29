# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import random
# data=pd.read_csv('Customer Purchase Data.csv')
# print(data.describe())
# print(data.info())
# dummies=pd.get_dummies(data,columns=['Age'],dtype='int64')
# print(dummies)
# data1=dummies.drop(columns=['Number'])
# print(data1)
# print(data1.columns)
# print(data1.corr().to_string())
# X=data1[["Income", "Membership_Years"]]
# Y=data1['Purchase_Frequency']
# print(X)
# print(Y)
# sns.scatterplot(data=data1,x='Income',y='Purchase_Frequency')
# plt.plot(X,Y)
# plt.show()
# random.seed(1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# regr= LinearRegression()
# regr.fit(X_train, Y_train)
# print(regr.score(X_test, Y_test))




# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import random
# data=pd.read_csv('Customer Purchase Data.csv')

# dummies=pd.get_dummies(data,columns=['Age'],dtype='int64')
# print(dummies)
# data1=dummies.drop(columns=['Number'])

# print(data1.columns)

# X=data1[['Membership_Years']]
# Y=data1['Purchase_Frequency']

# random.seed(1)
# X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)
# model=LogisticRegression()
# model.fit(X_train,Y_train)
# Y_pred=model.predict(X_test)
# accuracy=accuracy_score(Y_test,Y_pred)
# print('Model Score',model.score(X_train,Y_train))
# print('Accuracy: {:1.2f}%'.format(accuracy*100))

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor
import random

data=pd.read_csv('Customer Purchase Data.csv')
dummies=pd.get_dummies(data,columns=['Number'],dtype='int64')
data1=dummies.drop(columns=['Spending_Score'])
print(data.columns)
x=data1[['Membership_Years']]
y=data1['Purchase_Frequency']
sns.scatterplot(x='Income',y="Purchase_Frequency",data=data1)
random.seed(1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
knn=KNeighborsRegressor()
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print('KNN score ',knn.score(x_test,y_test))

plt.show()