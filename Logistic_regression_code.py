import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
data=pd.read_csv('Customer Purchase Data.csv')

dummies=pd.get_dummies(data,columns=['Age'],dtype='int64')
print(dummies)
data1=dummies.drop(columns=['Number'])
print(data1.columns)

X=data1[['Membership_Years']]
Y_continuous=data1['Purchase_Frequency']
threshold= Y_continuous.mean()
Y=(Y_continuous>threshold).astype(int)

random.seed(1)
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
accuracy=accuracy_score(Y_test,Y_pred)

print('Model Score',model.score(X_train,Y_train))
print('Accuracy: {:.2f}%'.format(accuracy*100))