# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')

'''
Dividing dataset into independent and dependent part.
'''

'''
Linear Regression : y=b0+b1.x
ordered squares: sum[(y-y')^2] -> min

'''


#first colon: extract rows 
#second colon: extract columns (-1 indicates ignore last column) 
#X independent variable (experience)
#Y dependent variable (salary)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values


#Split into train and test(we choose test size as 1/3 becz data set size is 30)
#So test data=10 and train data=20

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


#Fit Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting Test Set
Y_pred=regressor.predict(X_test)

#Visualize results (Training Set)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualize results (Test Set)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


#Feature Scaling:
'''
1. Standardization:    
    xstnd=(x-mean(x))/stndDeviation(x)
2. Normalisation:    
    xnorm=x-mean(x)/max(x)-min(x)    
'''
'''
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
'''  
 