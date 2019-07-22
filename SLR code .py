#Simple Linear Regression 
# DATA PREPROCESSING
#importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing Data set
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting into Train & Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 1/3, random_state = 0 )

"""Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)"""

#FITTING SLR into train set 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

#Predicting Test set results 
y_pred = reg.predict(x_test)

#Vizualize Train
plt.scatter(x_train, y_train, color = 'red' ) #Actual X and Y values 
plt.plot(x_train, reg.predict(x_train), color = 'blue') #our Regression line
plt.title('Salary vs Experience(Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Vizulaize Test
plt.scatter(x_test, y_test, color = 'red' ) #Actual X and Y test values 
plt.plot(x_train, reg.predict(x_train), color = 'blue') #our Regression line remains same
plt.title('Salary vs Experience(Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

