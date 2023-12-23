#Importing libraries
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# import data set
housing = pd.read_csv('Housing.csv')

# Data Preparation

housing.head()

#Convert 'yes' and 'no' into binary variable
varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})
# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Check the housing dataframe now
housing.head()

#Dummy Variables
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'

status = pd.get_dummies(housing['furnishingstatus'])
status.head()

# Let's drop the first column from status df using 'drop_first = True'
status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)

status.head()

# Add the results to the original housing dataframe
housing = pd.concat([housing, status], axis = 1)

# Now let's see the head of our dataframe.
housing.head()

housing = housing.drop('furnishingstatus',axis=1)

#housing.head()

# Splitting the Data into Training and Testing Sets

from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

# Rescaling the Features

#We will use MinMax scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()

#Heatmap to see correlation between variables
plt.figure(figsize=(25, 20))
sns.heatmap(df_train.corr(), cmap='YlGnBu', annot = True)
plt.show()

# Dividing into X and Y sets for the model building

y_train = df_train.pop('price')
X_train = df_train

# Building our model by RFE ( Recursive feature elimination)

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

housing.head()

# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)



rfe= RFE(lm)

rfe = RFE(lm, n_features_to_select=10)             # running RFE


rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col = X_train.columns[rfe.support_]
col

X_train.columns[~rfe.support_]



# Building Model using Statsmodel

Train the Model

import statsmodels.api as sm  
# Creating X_train dataframe with RFE selected variables
X_train_rfe = X_train[col]

# Adding  a constant variable 


X_train_rfe = sm.add_constant(X_train_rfe)

 # Running the linear model
lm = sm.OLS(y_train,X_train_rfe).fit()  

#Let's see the summary of our linear model
print(lm.summary())

# basement is not significant hence dropping this
X_train_new = X_train_rfe.drop(["basement"], axis = 1)

#building new model without basement
#add constant
X_train_lm = sm.add_constant(X_train_new)

X_train_new.head()

 # Running the linear model
lm = sm.OLS(y_train,X_train_new).fit()  

#Let's see the summary of our linear model
print(lm.summary())

X_train_new = X_train_new.drop(['const'], axis=1)

# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF']=[variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)



vif

X_train_new = X_train_new.drop(['bathrooms'], axis=1)

 #Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF']=[variance_inflation_factor(X_train_new.values,i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

Rebuilding the model without bathrooms

X_train_lm = sm.add_constant(X_train_new)

X_train_new.head()

lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model

#Let's see the summary of our linear model
print(lm.summary())

# Residual Analysis of the train data¶

# Residual Analysis 
y_train_pred = lm.predict(X_train_lm)
y_train_pred.head()

res = y_train - y_train_pred
sns.distplot(res)

# Make Prediction On Test Set

Making Prediction on Test Data Test on the basis of the learning from train data set

# Dividing into X_test and y_test¶



num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_test[num_vars] = scaler.transform(df_test[num_vars])

y_test = df_test.pop('price')
X_test = df_test

# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

# Making predictions
y_pred = lm.predict(X_test_new)

# Model Evaluation

# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)  


