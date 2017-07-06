import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
 
'''
Dividing dataset into independent and dependent part.
'''

'''
y=b0+b1(x1^1)+b2(x1^2)+b3(x1^3)+...+bn(x1^n)
'''

#used for curve fitting(parabolic feature)



#first colon: extract rows 
#second colon: extract columns (-1 indicates ignore last column) 
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values



#Split into train and test
'''
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
'''

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
linear_reg1=LinearRegression()
linear_reg1.fit(X,Y)


#Fitting Polynomial Regression to dataset  
from sklearn.preprocessing import PolynomialFeatures

#polynomial features is used for providing poly terms to x part 
poly_reg=PolynomialFeatures(degree=4)
#transforming feature var X into X_Poly 
#having og independent var positional level and its associate poly terms
X_poly=poly_reg.fit_transform(X)   

linear_reg2=LinearRegression()
linear_reg2.fit(X_poly,Y)

#visualising linear regression results
plt.scatter(X,Y,color="red")
plt.plot(X,linear_reg1.predict(X),color="green")
plt.title("Linear Regression Results. Truth/Bluff Detector")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#visualising polynomial regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color="red")
plt.plot(X_grid,linear_reg2.predict(poly_reg.fit_transform(X_grid)),color="green")
plt.title("Polynomial Regression Results. Truth/Bluff Detector")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Predicting new result with linear regression
linear_reg1.predict(6.5)

#Predicting new result with polynomial regression
linear_reg2.predict(poly_reg.fit_transform(6.5))