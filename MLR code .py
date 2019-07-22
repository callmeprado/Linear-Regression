#Multi Linear Regression 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing Data set
dataset = pd.read_csv('50_startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Categorical Datta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Var trap 
"""We don't need to do this manually as Python Libraries will
    automatically take care of this trap, but let's do it anyway"""
X = X[:,1:] #removing the first column of X, avoid duplication of variables 

#Splitting into Train & Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0 )

#Fitting MLR to the train set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train) 

#Predicting Test set results 
y_pred = reg.predict(x_test)

#Building Optimal Model by using Backward Elimination method 
import statsmodels.formula.api as sm 
 """We need to add an x0 to the b0 coefficient in the MLR formula, 
 for which add an another column setting x0 = 1"""
 X = np.append(arr=  np.ones((50,1)).astype(int), values =X, axis = 1 )
 
 """Now we make another array which will be called the optimal X"""
 
 X_opt = X[:, [0, 1,2,3,4, 5]] #using this we apply it on the NEW regressor 
 reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 reg_OLS.summary()
 
 X_opt = X[:, [0, 1,3,4, 5]] #using this we apply it on the NEW regressor 
 reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 reg_OLS.summary()

 X_opt = X[:, [0,3,5]] #using this we apply it on the NEW regressor 
 reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 reg_OLS.summary()
 
 X_opt = X[:, [0,3]] #using this we apply it on the NEW regressor 
 reg_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
 reg_OLS.summary()
 
 """ We have rigorously followed backward elimination method, and eliminated all P val variables (> 0.05), 
 hence, we have reached our optimal model. Leaving x1 (R&D investment) to be the only significant
 independent variable."""