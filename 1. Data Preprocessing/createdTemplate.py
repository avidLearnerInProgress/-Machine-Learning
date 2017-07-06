import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')

'''
Dividing dataset into independent and dependent part.
'''

#first colon: extract rows 
#second colon: extract columns (-1 indicates ignore last column) 
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values


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
 