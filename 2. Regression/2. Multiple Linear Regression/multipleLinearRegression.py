# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 22:50:02 2017

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')

'''
y=b0x+b1x1+b2x2+...+bnxn

'''
'''
Dividing dataset into independent and dependent part.
'''



#first colon: extract rows 
#second colon: extract columns (-1 indicates ignore last column) 
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#LabelEncoder for encoding categorical data
#OneHotEncoder for removing relational order (creating dummy variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

'''Avoiding dummy variable trap
removed first column from x 
start doing from second column!
'''
X=X[:,1:] 

#Split into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


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

#Multiple Linear Regression To training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting Test Set Results
y_pred=regressor.predict(X_test)


'''
Backward Elimination steps
#select significance level to stay in model 
# fit full model with all possible predictors
# consider predictor with highest p-value
#remove predictor
#fit model without this variable
'''

#Building optimal method using backward elimination
import statsmodels.api as sm
'''to satisfy the equation b0 + b1x1 + b2x2 +...
here we need to include b0 as 1 because statsmodels.api doesnt have b0 set by default
if not added sm doesnt consider b0 !! 
'''
#adding matrix of features x to one column (values->arr)
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1) #ones(no of lines,no of cols)
#first add all independent var and then remove some statistically 
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
#lower the p value more significant is the dependent variable is wrt independent var
#remove the independent var having highest p value

X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()


X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()


X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()


X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
 
#thus, we conclude that the independent var having strongest impact on the profit prediction is Rnd variable



