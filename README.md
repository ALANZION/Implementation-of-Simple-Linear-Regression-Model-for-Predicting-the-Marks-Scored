# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas .
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ALAN ZION H
RegisterNumber:212223240004
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:
## df.head()
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/91028d89-81e0-44db-ad71-ab9bf5010bed)
## df.tail()
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/8632af0d-c01c-4406-aefb-402487e29707)
## Values of X:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/0e590d92-3e58-44a5-8db8-6360d3e6d120)
## Values of Y:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/0ac0ca1f-2e27-47ca-b154-e4b28ff6d075)
## Values of Y prediction:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/3c612329-75d2-48c9-83af-88c466575bfb)
## Values of Y test:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/041adcb6-90bc-4322-b51d-c8ff95d23241)
## Training set graph:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/35fd9a06-e9b3-49c2-a950-5810f97cfd8d)
## Test set graph:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/62ba711f-8a44-404c-b3b1-66b03b2eb1af)
## Value of MSE,MAE & RMSE:
![image](https://github.com/ALANZION/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145743064/0a9c6b37-caa4-47fe-95a2-3b8a042978db)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
