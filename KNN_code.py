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
