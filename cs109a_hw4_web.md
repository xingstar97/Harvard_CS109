---
title: H4
notebook: cs109a_hw4_web.ipynb
nav_include: 6
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)
```





<style>
blockquote { background: #AEDE94; }
h1 { 
    padding-top: 25px;
    padding-bottom: 25px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
h2 { 
    padding-top: 10px;
    padding-bottom: 10px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}

div.exercise {
	background-color: #ffcccc;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
}

span.sub-q {
	font-weight: bold;
}
div.theme {
	background-color: #DDDDDD;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 18pt;
}
div.gc { 
	background-color: #AEDE94;
	border-color: #E9967A; 	 
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 12pt;
}
p.q1 { 
    padding-top: 5px;
    padding-bottom: 5px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}
header {
   padding-top: 35px;
    padding-bottom: 35px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
</style>







```python
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from pandas.core import datetools
%matplotlib inline

import seaborn as sns
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)
```


In this homework, we will focus on regularization and cross validation. We will continue to build regression models for the [Capital Bikeshare program](https://www.capitalbikeshare.com) in Washington D.C.  See homework 3 for more information about the Capital Bikeshare data that we'll be using extensively. 

## Data pre-processing

**1.1** Read in the provided `bikes_student.csv` to a data frame named `bikes_main`. Split it into a training set `bikes_train` and a validation set `bikes_val`. Use `random_state=90`, a test set size of .2, and stratify on month. Remember to specify the data's index column as you read it in.



```python
bikes_main = pd.read_csv("data/bikes_student.csv", index_col=0).reset_index(drop=True)
display(bikes_main.head())

bikes_train, bikes_val = train_test_split(bikes_main, test_size=.2, random_state=90, stratify=bikes_main['month'])
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dteday</th>
      <th>hour</th>
      <th>year</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>counts</th>
      <th>Feb</th>
      <th>Mar</th>
      <th>Apr</th>
      <th>May</th>
      <th>Jun</th>
      <th>Jul</th>
      <th>Aug</th>
      <th>Sept</th>
      <th>Oct</th>
      <th>Nov</th>
      <th>Dec</th>
      <th>spring</th>
      <th>summer</th>
      <th>fall</th>
      <th>Mon</th>
      <th>Tue</th>
      <th>Wed</th>
      <th>Thu</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Cloudy</th>
      <th>Snow</th>
      <th>Storm</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-09-07</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.64</td>
      <td>0.5758</td>
      <td>0.89</td>
      <td>0.0000</td>
      <td>14</td>
      <td>212</td>
      <td>226</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-03-21</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.52</td>
      <td>0.5000</td>
      <td>0.83</td>
      <td>0.0896</td>
      <td>4</td>
      <td>22</td>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-08-16</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.70</td>
      <td>0.6515</td>
      <td>0.54</td>
      <td>0.1045</td>
      <td>58</td>
      <td>168</td>
      <td>226</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-04-28</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.62</td>
      <td>0.5758</td>
      <td>0.83</td>
      <td>0.2985</td>
      <td>18</td>
      <td>103</td>
      <td>121</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-01-04</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.08</td>
      <td>0.0606</td>
      <td>0.42</td>
      <td>0.3284</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


**1.2** As with last homework, the response will be the `counts` column and we'll drop `counts`, `registered` and `casual` for being trivial predictors, drop `workingday` and `month` for being multicolinear with other columns, and `dteday` for being inappropriate for regression. Write code to do this.

Encapsulate this process as a function with appropriate inputs and outputs, and test your code by producing `practice_y_train` and `practice_X_train`




```python
def process(df, columns_to_drop, response):    
    y_df = df[response]
    X_df = df.drop(columns_to_drop, axis=1)
    
    return y_df, X_df

columns_to_drop = ['counts', 'registered', 'casual', 'workingday', 'month', 'dteday']
reponse_variable = 'counts'
practice_y_train, practice_X_train = process(bikes_train, columns_to_drop, reponse_variable)
```


**1.3** Write a function to standardize a provided subset of columns in your training/validation/test sets. Remember that while you will be scaling all of your data, you must learn the scaling parameters (mean and SD) from only the training set.

Test your code by building a list of all non-binary columns in your `practice_X_train` and scaling only those columns. Call the result `practice_X_train_scaled`. Display the `.describe()` and verify that you have correctly scaled all columns, including the polynomial columns.

**Hint: employ the provided list of binary columns and use `pd.columns.difference()`**

`binary_columns = ['holiday', 'workingday', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'spring', 'summer', 'fall', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Cloudy', 'Snow', 'Storm']`



```python
def standardize(df, target_column, mean=np.inf, std=np.inf):
    if target_column in df.columns: 
        df_scaled = df 
    
        if mean == np.inf:
            mean = np.mean(df_scaled[target_column])
            
        if std == np.inf: 
            std = np.std(df_scaled[target_column])
            
        df_scaled[target_column] = df_scaled[target_column].apply(lambda x: (x - mean) / std) 
        
        return df_scaled, mean, std
    
    return

binary_columns = ['holiday', 'workingday', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 'spring', 
                  'summer', 'fall', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 
                  'Sat', 'Cloudy', 'Snow', 'Storm']
nonbinary_columns = practice_X_train.columns.difference(binary_columns)

## There are only two unique value for column 'year', so we can treat it as a binary column
nonbinary_columns = nonbinary_columns.difference(['year'])
practice_X_train_scaled = practice_X_train.copy()

for target_column in nonbinary_columns:
    practice_X_train_scaled, target_column_mean, target_column_stdev = standardize(
        practice_X_train_scaled, target_column)
```


**1.4** Write a code to augment your a dataset with higher-order features for `temp`, `atemp`, `hum`,`windspeed`, and `hour`. You should include ONLY pure powers of these columns. So with degree=2 you should produce `atemp^2` and `hum^2` but not `atemp*hum` or any other two-feature interactions. 


Encapsulate this process as a function with apropriate inputs and outputs, and test your code by producing `practice_X_train_poly`, a training dataset with qudratic and cubic features built from `practice_X_train_scaled`, and printing `practice_X_train_poly`'s column names and `.head()`.



```python
def poly(df, target_column, degree):
    tra = PolynomialFeatures(degree, include_bias=False)
    array_poly = tra.fit_transform(df[target_column].values.reshape(-1,1))
    df_poly = pd.DataFrame(array_poly)
    
    for i in range(degree): 
        if i == 0: 
            df_poly = df_poly.rename(columns={i: '%s' % target_column})
        else: 
            df_poly = df_poly.rename(columns={i: '%s_%i' % (target_column, (i+1))})
    
    return df_poly

practice_X_train_poly = practice_X_train_scaled.copy().reset_index(drop=True)
ploy_columns = ['temp', 'atemp', 'hum', 'windspeed', 'hour']
practice_X_train_poly = practice_X_train_poly.drop(ploy_columns, axis=1)

for target_column in ploy_columns:
    df_poly = poly(practice_X_train_scaled, target_column, 3) 
    practice_X_train_poly = practice_X_train_poly.merge(df_poly, left_index=True, right_index=True)
```


**1.5** Write code to add interaction terms to the model. Specifically, we want interactions between the continuous predictors (`temp`,`atemp`, `hum`,`windspeed`) and the month and weekday dummies (`Feb`, `Mar`...`Dec`, `Mon`, `Tue`, ... `Sat`). That means you SHOULD build `atemp*Feb` and `hum*Mon` and so on, but NOT `Feb*Mar` and NOT `Feb*Tue`. The interaction terms should always be a continuous feature times a month dummy or a continuous feature times a weekday dummy.


Encapsulate this process as a function with appropriate inputs and outputs, and test your code by adding interaction terms to `practice_X_train_poly` and show its column names and `.head()`**




```python
def interaction(df, a, b):
    df["%s_%s" % (a, b)] = df[a] * df[b]
    return df

continuous_columns = ['temp', 'atemp', 'hum', 'windspeed']
month_week_columns = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 
                      'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

for c in continuous_columns:
    for mw in month_week_columns:
        interaction(practice_X_train_poly, c, mw)
```


**1.6** Combine all your code so far into a function that takes in `bikes_train`, `bikes_val`, the names of columns for polynomial, the target column, the columns to be dropped and produces computation-ready design matrices `X_train` and `X_val` and responses `y_train` and `y_val`. Your final function should build correct, scaled design matrices with the stated interaction terms and any polynomial degree.



```python
def get_design_mats(train_df, val_df, degree, 
                    columns_forpoly=['temp', 'atemp', 'hum', 'windspeed', 'hour'],
                    target_col='counts', 
                    bad_columns=['counts', 'registered', 'casual', 'workingday', 'month', 'dteday']):                                    
    ## Step 1 - Process Bad Columns 
    y_train, X_train_process = process(train_df, bad_columns, target_col)
    y_val, X_val_process = process(val_df, bad_columns, target_col)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    
    ## Step 2 - Standardize Bad Columns 
    binary_columns = ['holiday', 'workingday', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 
                      'Sept', 'Oct', 'Nov', 'Dec', 'spring', 'summer', 'fall', 'Mon', 'Tue', 
                      'Wed', 'Thu', 'Fri', 'Sat', 'Cloudy', 'Snow', 'Storm']

    X_train_nonbinary_columns = X_train_process.columns.difference(binary_columns)
    X_val_nonbinary_columns = X_val_process.columns.difference(binary_columns)
    
    if 'year' in X_train_nonbinary_columns: 
        ## Since there are only two values for column 'year', we can treat it as binary column
        X_train_nonbinary_columns = X_train_nonbinary_columns.difference(['year'])

    if 'year' in X_val_nonbinary_columns: 
        ## Since there are only two values for column 'year', we can treat it as binary column
        X_val_nonbinary_columns = X_val_nonbinary_columns.difference(['year'])

    X_train_scaled = X_train_process.copy()
    X_val_scaled = X_val_process.copy()
    for target_column in X_train_nonbinary_columns:
        X_train_scaled, X_train_column_mean, X_train_column_stdev = standardize(X_train_scaled, target_column)
        X_val_scaled, _, _ = standardize(X_val_scaled, target_column, X_train_column_mean, X_train_column_stdev)
    
    ## Step 3 - Add polynomial terms 
    X_train_poly = X_train_scaled.copy().reset_index(drop=True)
    X_train_poly = X_train_poly.drop(columns_forpoly, axis=1)
    
    X_val_poly = X_val_scaled.copy().reset_index(drop=True)
    X_val_poly = X_val_poly.drop(columns_forpoly, axis=1)

    for target_column in columns_forpoly:
        X_train_column_poly = poly(X_train_scaled, target_column, degree) 
        X_train_poly = X_train_poly.merge(X_train_column_poly, left_index=True, right_index=True)    
        
        X_val_column_poly = poly(X_val_scaled, target_column, degree) 
        X_val_poly = X_val_poly.merge(X_val_column_poly, left_index=True, right_index=True)    
    
    ## Step 4 - Add interaction terms
    continuous_columns = ['temp', 'atemp', 'hum', 'windspeed']
    month_week_columns = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', 
                          'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    
    x_train = X_train_poly.copy()
    x_val = X_val_poly.copy()
    for c in continuous_columns:
        for mw in month_week_columns:
            interaction(x_train, c, mw)
            interaction(x_val, c, mw)
    
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val \
= get_design_mats(bikes_train, bikes_val, degree=3, 
                  columns_forpoly=['temp', 'atemp', 'hum','windspeed', 'hour'],
                  target_col='counts', 
                  bad_columns=['counts', 'registered', 'casual', 'workingday', 'month', 'dteday'])
display(x_train.describe())
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>holiday</th>
      <th>Feb</th>
      <th>Mar</th>
      <th>Apr</th>
      <th>May</th>
      <th>Jun</th>
      <th>Jul</th>
      <th>Aug</th>
      <th>Sept</th>
      <th>Oct</th>
      <th>Nov</th>
      <th>Dec</th>
      <th>spring</th>
      <th>summer</th>
      <th>fall</th>
      <th>Mon</th>
      <th>Tue</th>
      <th>Wed</th>
      <th>Thu</th>
      <th>Fri</th>
      <th>Sat</th>
      <th>Cloudy</th>
      <th>Snow</th>
      <th>Storm</th>
      <th>temp</th>
      <th>temp_2</th>
      <th>temp_3</th>
      <th>atemp</th>
      <th>atemp_2</th>
      <th>atemp_3</th>
      <th>hum</th>
      <th>hum_2</th>
      <th>hum_3</th>
      <th>windspeed</th>
      <th>windspeed_2</th>
      <th>windspeed_3</th>
      <th>hour</th>
      <th>hour_2</th>
      <th>hour_3</th>
      <th>temp_Feb</th>
      <th>temp_Mar</th>
      <th>temp_Apr</th>
      <th>temp_May</th>
      <th>temp_Jun</th>
      <th>temp_Jul</th>
      <th>temp_Aug</th>
      <th>temp_Sept</th>
      <th>temp_Oct</th>
      <th>temp_Nov</th>
      <th>temp_Dec</th>
      <th>temp_Mon</th>
      <th>temp_Tue</th>
      <th>temp_Wed</th>
      <th>temp_Thu</th>
      <th>temp_Fri</th>
      <th>temp_Sat</th>
      <th>atemp_Feb</th>
      <th>atemp_Mar</th>
      <th>atemp_Apr</th>
      <th>atemp_May</th>
      <th>atemp_Jun</th>
      <th>atemp_Jul</th>
      <th>atemp_Aug</th>
      <th>atemp_Sept</th>
      <th>atemp_Oct</th>
      <th>atemp_Nov</th>
      <th>atemp_Dec</th>
      <th>atemp_Mon</th>
      <th>atemp_Tue</th>
      <th>atemp_Wed</th>
      <th>atemp_Thu</th>
      <th>atemp_Fri</th>
      <th>atemp_Sat</th>
      <th>hum_Feb</th>
      <th>hum_Mar</th>
      <th>hum_Apr</th>
      <th>hum_May</th>
      <th>hum_Jun</th>
      <th>hum_Jul</th>
      <th>hum_Aug</th>
      <th>hum_Sept</th>
      <th>hum_Oct</th>
      <th>hum_Nov</th>
      <th>hum_Dec</th>
      <th>hum_Mon</th>
      <th>hum_Tue</th>
      <th>hum_Wed</th>
      <th>hum_Thu</th>
      <th>hum_Fri</th>
      <th>hum_Sat</th>
      <th>windspeed_Feb</th>
      <th>windspeed_Mar</th>
      <th>windspeed_Apr</th>
      <th>windspeed_May</th>
      <th>windspeed_Jun</th>
      <th>windspeed_Jul</th>
      <th>windspeed_Aug</th>
      <th>windspeed_Sept</th>
      <th>windspeed_Oct</th>
      <th>windspeed_Nov</th>
      <th>windspeed_Dec</th>
      <th>windspeed_Mon</th>
      <th>windspeed_Tue</th>
      <th>windspeed_Wed</th>
      <th>windspeed_Thu</th>
      <th>windspeed_Fri</th>
      <th>windspeed_Sat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.0</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1.000000e+03</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.509000</td>
      <td>0.027000</td>
      <td>0.078000</td>
      <td>0.085000</td>
      <td>0.082000</td>
      <td>0.086000</td>
      <td>0.08300</td>
      <td>0.086000</td>
      <td>0.085000</td>
      <td>0.082000</td>
      <td>0.08300</td>
      <td>0.082000</td>
      <td>0.086000</td>
      <td>0.255000</td>
      <td>0.258000</td>
      <td>0.248000</td>
      <td>0.143000</td>
      <td>0.148000</td>
      <td>0.162000</td>
      <td>0.128000</td>
      <td>0.12700</td>
      <td>0.15000</td>
      <td>0.280000</td>
      <td>0.082000</td>
      <td>0.0</td>
      <td>-4.038325e-15</td>
      <td>1.000000</td>
      <td>-0.027791</td>
      <td>4.026779e-15</td>
      <td>1.000000</td>
      <td>-0.105907</td>
      <td>-4.013789e-15</td>
      <td>1.000000</td>
      <td>-0.180672</td>
      <td>8.927858e-15</td>
      <td>1.000000</td>
      <td>0.665403</td>
      <td>-1.811190e-16</td>
      <td>1.000000</td>
      <td>0.042444</td>
      <td>-0.084718</td>
      <td>-0.049297</td>
      <td>-0.008857</td>
      <td>0.043253</td>
      <td>0.081930</td>
      <td>0.113156</td>
      <td>0.093621</td>
      <td>0.051193</td>
      <td>-0.004360</td>
      <td>-0.054595</td>
      <td>-0.076640</td>
      <td>-0.004603</td>
      <td>0.007512</td>
      <td>0.001398</td>
      <td>0.017134</td>
      <td>0.002473</td>
      <td>-0.005792</td>
      <td>-0.084275</td>
      <td>-0.048254</td>
      <td>-0.006756</td>
      <td>0.044350</td>
      <td>0.080102</td>
      <td>0.112129</td>
      <td>0.084750</td>
      <td>0.050252</td>
      <td>-0.000850</td>
      <td>-0.050854</td>
      <td>-0.073364</td>
      <td>-0.001580</td>
      <td>0.009120</td>
      <td>0.000842</td>
      <td>0.017054</td>
      <td>-0.003612</td>
      <td>-0.006207</td>
      <td>-0.016993</td>
      <td>-0.009814</td>
      <td>-0.019378</td>
      <td>0.027735</td>
      <td>-0.030795</td>
      <td>-0.013424</td>
      <td>0.006119</td>
      <td>0.024119</td>
      <td>0.033308</td>
      <td>-0.002914</td>
      <td>0.033365</td>
      <td>0.002640</td>
      <td>0.008277</td>
      <td>-0.003972</td>
      <td>-0.000301</td>
      <td>-0.000462</td>
      <td>0.000101</td>
      <td>0.006143</td>
      <td>0.018765</td>
      <td>0.023665</td>
      <td>0.004863</td>
      <td>0.003238</td>
      <td>-0.017571</td>
      <td>-0.011146</td>
      <td>-0.014081</td>
      <td>-0.014803</td>
      <td>0.001709</td>
      <td>-0.017209</td>
      <td>-0.013921</td>
      <td>-0.013266</td>
      <td>0.004386</td>
      <td>-0.011268</td>
      <td>0.001801</td>
      <td>0.017336</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500169</td>
      <td>0.162164</td>
      <td>0.268306</td>
      <td>0.279021</td>
      <td>0.274502</td>
      <td>0.280504</td>
      <td>0.27602</td>
      <td>0.280504</td>
      <td>0.279021</td>
      <td>0.274502</td>
      <td>0.27602</td>
      <td>0.274502</td>
      <td>0.280504</td>
      <td>0.436079</td>
      <td>0.437753</td>
      <td>0.432068</td>
      <td>0.350248</td>
      <td>0.355278</td>
      <td>0.368635</td>
      <td>0.334257</td>
      <td>0.33314</td>
      <td>0.35725</td>
      <td>0.449224</td>
      <td>0.274502</td>
      <td>0.0</td>
      <td>1.000500e+00</td>
      <td>1.010902</td>
      <td>2.365524</td>
      <td>1.000500e+00</td>
      <td>1.041730</td>
      <td>2.456716</td>
      <td>1.000500e+00</td>
      <td>1.115265</td>
      <td>2.759186</td>
      <td>1.000500e+00</td>
      <td>1.769451</td>
      <td>6.585134</td>
      <td>1.000500e+00</td>
      <td>0.897533</td>
      <td>1.969596</td>
      <td>0.323573</td>
      <td>0.230346</td>
      <td>0.160060</td>
      <td>0.202329</td>
      <td>0.305387</td>
      <td>0.385865</td>
      <td>0.326785</td>
      <td>0.218353</td>
      <td>0.179759</td>
      <td>0.228052</td>
      <td>0.283552</td>
      <td>0.387803</td>
      <td>0.373567</td>
      <td>0.413001</td>
      <td>0.363481</td>
      <td>0.346771</td>
      <td>0.396963</td>
      <td>0.327661</td>
      <td>0.240164</td>
      <td>0.166432</td>
      <td>0.202331</td>
      <td>0.295116</td>
      <td>0.385321</td>
      <td>0.315576</td>
      <td>0.216298</td>
      <td>0.187281</td>
      <td>0.223262</td>
      <td>0.278553</td>
      <td>0.381743</td>
      <td>0.375285</td>
      <td>0.406983</td>
      <td>0.360264</td>
      <td>0.347039</td>
      <td>0.404316</td>
      <td>0.289796</td>
      <td>0.337607</td>
      <td>0.347008</td>
      <td>0.263407</td>
      <td>0.283426</td>
      <td>0.263061</td>
      <td>0.275895</td>
      <td>0.264578</td>
      <td>0.262246</td>
      <td>0.269179</td>
      <td>0.282560</td>
      <td>0.381031</td>
      <td>0.379798</td>
      <td>0.407580</td>
      <td>0.338340</td>
      <td>0.355459</td>
      <td>0.398312</td>
      <td>0.329464</td>
      <td>0.319963</td>
      <td>0.305876</td>
      <td>0.240924</td>
      <td>0.253669</td>
      <td>0.278028</td>
      <td>0.297143</td>
      <td>0.242928</td>
      <td>0.281089</td>
      <td>0.288773</td>
      <td>0.310498</td>
      <td>0.370430</td>
      <td>0.346023</td>
      <td>0.403203</td>
      <td>0.312060</td>
      <td>0.376432</td>
      <td>0.428265</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-2.347976e+00</td>
      <td>0.001402</td>
      <td>-12.944365</td>
      <td>-2.402605e+00</td>
      <td>0.000275</td>
      <td>-13.869057</td>
      <td>-3.397602e+00</td>
      <td>0.000002</td>
      <td>-39.220902</td>
      <td>-1.554205e+00</td>
      <td>0.000128</td>
      <td>-3.754263</td>
      <td>-1.646163e+00</td>
      <td>0.002152</td>
      <td>-4.460859</td>
      <td>-1.933121</td>
      <td>-1.621979</td>
      <td>-0.999697</td>
      <td>-0.792269</td>
      <td>-0.066273</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.273701</td>
      <td>-1.414552</td>
      <td>-1.518266</td>
      <td>-1.725693</td>
      <td>-2.347976</td>
      <td>-1.933121</td>
      <td>-2.244262</td>
      <td>-1.829407</td>
      <td>-2.036834</td>
      <td>-1.933121</td>
      <td>-2.137233</td>
      <td>-1.695726</td>
      <td>-1.076915</td>
      <td>-0.900195</td>
      <td>-0.016596</td>
      <td>0.000000</td>
      <td>-1.342286</td>
      <td>-0.193316</td>
      <td>-1.607074</td>
      <td>-1.430354</td>
      <td>-1.783794</td>
      <td>-2.402605</td>
      <td>-2.137233</td>
      <td>-2.137233</td>
      <td>-1.960514</td>
      <td>-2.049165</td>
      <td>-2.137233</td>
      <td>-2.016765</td>
      <td>-3.397602</td>
      <td>-2.547856</td>
      <td>-2.069874</td>
      <td>-2.122984</td>
      <td>-2.176093</td>
      <td>-2.016765</td>
      <td>-1.910547</td>
      <td>-1.538783</td>
      <td>-1.963656</td>
      <td>-1.485674</td>
      <td>-2.282311</td>
      <td>-2.282311</td>
      <td>-2.335420</td>
      <td>-3.397602</td>
      <td>-2.229202</td>
      <td>-2.547856</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
      <td>-1.554205</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>-7.922693e-01</td>
      <td>0.231484</td>
      <td>-0.497300</td>
      <td>-8.121270e-01</td>
      <td>0.136927</td>
      <td>-0.535638</td>
      <td>-7.421467e-01</td>
      <td>0.139237</td>
      <td>-0.408761</td>
      <td>-7.231056e-01</td>
      <td>0.061656</td>
      <td>-0.378099</td>
      <td>-9.189949e-01</td>
      <td>0.152028</td>
      <td>-0.776139</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.744066e-02</td>
      <td>0.751950</td>
      <td>0.000052</td>
      <td>7.147176e-02</td>
      <td>0.751693</td>
      <td>0.000365</td>
      <td>5.448995e-02</td>
      <td>0.632432</td>
      <td>0.000162</td>
      <td>-1.130295e-02</td>
      <td>0.491815</td>
      <td>-0.000001</td>
      <td>-4.639332e-02</td>
      <td>0.844552</td>
      <td>-0.000100</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>8.671507e-01</td>
      <td>1.457149</td>
      <td>0.652054</td>
      <td>8.670022e-01</td>
      <td>1.489478</td>
      <td>0.651719</td>
      <td>8.511266e-01</td>
      <td>1.621134</td>
      <td>0.616570</td>
      <td>4.634972e-01</td>
      <td>1.118505</td>
      <td>0.099573</td>
      <td>8.262083e-01</td>
      <td>1.593929</td>
      <td>0.563986</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>2.319143e+00</td>
      <td>5.512989</td>
      <td>12.473338</td>
      <td>2.546131e+00</td>
      <td>6.482785</td>
      <td>16.506023</td>
      <td>1.913309e+00</td>
      <td>11.543700</td>
      <td>7.004146</td>
      <td>5.211499e+00</td>
      <td>27.159723</td>
      <td>141.542871</td>
      <td>1.698810e+00</td>
      <td>2.885955</td>
      <td>4.902689</td>
      <td>0.348582</td>
      <td>0.867151</td>
      <td>1.282006</td>
      <td>1.800574</td>
      <td>2.215429</td>
      <td>2.319143</td>
      <td>2.111716</td>
      <td>1.696861</td>
      <td>1.178292</td>
      <td>0.556009</td>
      <td>0.141154</td>
      <td>2.111716</td>
      <td>2.008002</td>
      <td>2.319143</td>
      <td>2.215429</td>
      <td>2.215429</td>
      <td>2.008002</td>
      <td>0.336843</td>
      <td>0.867002</td>
      <td>1.043722</td>
      <td>1.927321</td>
      <td>2.369412</td>
      <td>2.546131</td>
      <td>1.927321</td>
      <td>1.750601</td>
      <td>1.220442</td>
      <td>0.867002</td>
      <td>0.160123</td>
      <td>2.192692</td>
      <td>1.927321</td>
      <td>2.369412</td>
      <td>2.546131</td>
      <td>2.369412</td>
      <td>1.927321</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>1.594654</td>
      <td>1.594654</td>
      <td>1.275999</td>
      <td>1.594654</td>
      <td>1.594654</td>
      <td>1.913309</td>
      <td>1.594654</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>1.594654</td>
      <td>1.913309</td>
      <td>1.913309</td>
      <td>3.668597</td>
      <td>3.668597</td>
      <td>4.143398</td>
      <td>2.007194</td>
      <td>2.125696</td>
      <td>5.211499</td>
      <td>2.837498</td>
      <td>2.600496</td>
      <td>2.600496</td>
      <td>2.837498</td>
      <td>3.312298</td>
      <td>4.143398</td>
      <td>2.600496</td>
      <td>2.600496</td>
      <td>3.668597</td>
      <td>3.312298</td>
      <td>3.668597</td>
    </tr>
  </tbody>
</table>
</div>


## Regularization via Ridge 

**2.1** For each degree in 1 through 8:

1.  Build the training design matrix and validation design matrix using the function `get_design_mats` with polynomial terms up through the specified degree.

2.  Fit a regression model to the training data.

3.  Report the model's score on the validation data.



```python
ols_r2_score = pd.DataFrame(index=range(1, 9), columns=['OLS'])

for d in range(1, 9):
    x_train, y_train, x_val, y_val = \
        get_design_mats(bikes_train, bikes_val, d,
                        columns_forpoly = ['temp', 'atemp', 'hum', 'windspeed', 'hour'], 
                        bad_columns = ['counts', 'registered', 'casual', 'workingday', 'month', 'dteday'])
    
    X_train = sm.add_constant(x_train)
    OLSModel = sm.OLS(y_train, X_train)
    ols = OLSModel.fit()
    
    X_test = sm.add_constant(x_val)
    ols_r2_score.loc[d, 'OLS'] = r2_score(y_val, ols.predict(X_test))
    
## Plot 
plt.figure(figsize=(8, 5))
plt.plot(ols_r2_score.index, ols_r2_score.OLS, '-o')
plt.ylabel('R-squared')
plt.xlabel('Degree of Polynomial Terms')
plt.title('Validation Set')
plt.show()
```



![png](cs109a_hw4_web_files/cs109a_hw4_web_19_0.png)


**2.2** Discuss patterns you see in the results from 2.1. Which model would you select, and why?**

In general, the higher degree of polynomial models, the higher R-squared in validation set, even though it's not a strict monotonic relationship. I would select the polynomial model with degree of 8, since it has the largest R-squared in validation set. 

**2.3** Let's try regularizing our models via ridge regression. Build a table showing the validation set $R^2$ of polynomial models with degree from 1-8, regularized at the levels $\lambda = (.01, .05, .1,.5, 1, 5, 10, 50, 100)$. Do not perform cross validation at this point, simply report performance on the single validation set. 




```python
## Train ridge
lambdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
ridge_r2_score = pd.DataFrame(index=range(1, 9), columns=lambdas)

for d in range(1, 9):
    for i, lam in enumerate(lambdas):
        x_train, y_train, x_val, y_val = \
            get_design_mats(bikes_train, bikes_val, d,
                            columns_forpoly = ['temp', 'atemp', 'hum', 'windspeed', 'hour'], 
                            bad_columns = ['counts', 'registered', 'casual', 'workingday', 'month', 'dteday'])

        ridge_reg = Ridge(alpha = lam) 
        ridge_reg.fit(x_train, y_train) 

        ridge_r2_score.loc[d, lam] = r2_score(y_val, ridge_reg.predict(x_val))
        
        
ridge_r2_score = ridge_r2_score.astype(float)

f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(ridge_r2_score, annot=True, linewidths=.5, ax=ax)
ax.set_ylabel('Degree of Polynomial Terms')
ax.set_xlabel('Lambda in Ridge Regression')
ax.set_title('R-squared on Validation Set');
```



![png](cs109a_hw4_web_files/cs109a_hw4_web_23_0.png)


**2.4** Find the best-scoring degree and regularization combination.



```python
ridge_r2_score.columns = ridge_r2_score.columns.map(str)
print("Best ridge model is with lambda of %s and polynomial degree of %s." % 
      (ridge_r2_score.max().idxmax(), str(ridge_r2_score[ridge_r2_score.max().idxmax()].idxmax())))
```


    Best ridge model is with lambda of 0.01 and polynomial degree of 8.


**2.5** It's time to see how well our selected model will do on future data. Read in the provided test dataset `data/bikes_test.csv`, do any required formatting, and report the best model's $R^2$ score. How does it compare to the validation set score that made us choose this model? 



```python
bikes_test = pd.read_csv("data/bikes_test.csv", index_col=0).reset_index(drop=True)

d = 8
lam = 0.01

x_train, y_train, x_test, y_test = \
    get_design_mats(bikes_main, bikes_test, d,
                    columns_forpoly = ['temp', 'atemp', 'hum', 'windspeed', 'hour'], 
                    bad_columns = ['counts', 'registered', 'casual', 'workingday', 'month', 'dteday'])

ridge_reg = Ridge(alpha = lam) 
ridge_reg.fit(x_train, y_train) 

ridge_r2_score_test = r2_score(y_test, ridge_reg.predict(x_test))
print("Best model's R-squared on test data set is %f." % ridge_r2_score_test)
```


    Best model's R-squared on test data set is 0.586768.


## Comparing Ridge, Lasso, and OLS

**3.1** Build a dataset with polynomial degree 1 and fit an OLS model, a Ridge model, and a Lasso model. Use `RidgeCV` and `LassoCV` to select the best regularization level from among `(.1,.5,1,5,10,50,100)`. 



```python
## Build the dataset 
d = 1
x_train, y_train, x_test, y_test = \
    get_design_mats(bikes_main, bikes_test, d,
                    columns_forpoly = ['temp', 'atemp', 'hum', 'windspeed', 'hour'], 
                    bad_columns = ['counts', 'registered', 'casual', 'workingday', 'month', 'dteday'])

lambdas = [0.1, 0.5, 1, 5, 10, 50, 100]

## OLS model
ols = LinearRegression()
ols.fit(x_train, y_train)

ols_r2_score_test = r2_score(y_test, ols.predict(x_test))
print("OLS Model R-squared is {:.3f}".format(ols_r2_score_test))
```


    OLS Model R-squared is 0.359




```python
## Ridge model with leave one out CV 
ridge = RidgeCV(alphas=lambdas, store_cv_values=True).fit(x_train, y_train)
ridge_cv_mse = list(np.mean(ridge.cv_values_, axis=0))
ridge_df = pd.DataFrame({'lambdas': lambdas, 'ridge_cv_mse': ridge_cv_mse})
print("The best lambda in Ridge Model is {:.2f}".format(ridge.alpha_))

## Ridge with optimal lambda
ridge_best_lam = ridge.alpha_
ridge_reg = Ridge(alpha = ridge_best_lam) 
ridge_reg.fit(x_train, y_train) 

ridge_r2_score_test = r2_score(y_test, ridge_reg.predict(x_test))
print("Ridge Model R-squared is {:.3f}".format(ridge_r2_score_test))

## Plot 
plt.figure(figsize=(8, 5))
plt.plot(ridge_df.lambdas, ridge_df.ridge_cv_mse, '-o')
plt.ylabel('MSE')
plt.xlabel('Lambdas in Ridge Regression')
plt.title('Cross Validation on Ridge Regression')
plt.show()
```


    The best lambda in Ridge Model is 50.00
    Ridge Model R-squared is 0.392



![png](cs109a_hw4_web_files/cs109a_hw4_web_31_1.png)




```python
## Lasso model with leave one out CV 
lasso = LassoCV(alphas=lambdas, cv=x_train.shape[0], max_iter=100000).fit(x_train, y_train)

df = pd.DataFrame(lasso.mse_path_)
lasso_cv_mse = df.mean(axis=1)
lasso_cv_mse = list(lasso_cv_mse)
lasso_cv_mse.reverse()
lasso_df = pd.DataFrame({'lambdas': lambdas, 'lasso_cv_mse': lasso_cv_mse})
print("The best lambda in Lasso Model is {:.2f}".format(lasso.alpha_))

## Lasso with optimal lambda
lasso_best_lam = lasso.alpha_
lasso_reg = Lasso(alpha = lasso_best_lam) 
lasso_reg.fit(x_train, y_train) 

lasso_r2_score_test = r2_score(y_test, lasso_reg.predict(x_test))
print("Lasso Model R-squared is {:.2f}".format(lasso_r2_score_test))

## Plot
plt.figure(figsize=(8, 5))
plt.plot(lasso_df.lambdas, lasso_df.lasso_cv_mse, '-o')
plt.ylabel('MSE')
plt.xlabel('Lambdas in Lasso Regression')
plt.title('Cross Validation on Lasso Regression')
plt.show()
```


    The best lambda in Lasso Model is 0.50
    Lasso Model R-squared is 0.38



![png](cs109a_hw4_web_files/cs109a_hw4_web_32_1.png)




```python
## Model Comparison
print("OLS Model R-squared is %f." % ols_r2_score_test)
print("Lasso Model R-squared is %f." % lasso_r2_score_test)
print("Ridge Model R-squared is %f." % ridge_r2_score_test)
```


    OLS Model R-squared is 0.358744.
    Lasso Model R-squared is 0.381019.
    Ridge Model R-squared is 0.392292.



**3.2** Plot histograms of the coefficients found by each of OLS, ridge, and lasso. What trends do you see in the magnitude of the coefficients?



```python
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(ols.coef_, 20, alpha=0.3, label="OLS")
ax.hist(ridge_reg.coef_, 20, alpha=0.6, label="Ridge")
ax.hist(lasso_reg.coef_, 20, alpha=0.9, label="Lasso")
ax.legend(prop={'size': 12});
```



![png](cs109a_hw4_web_files/cs109a_hw4_web_35_0.png)


Patterns: 
- OLS overfits the model that some coefficients of the independent variables are very large, either positive or negative. 
- Lasso has a lot of variables with mutted coefficient, which shows as the peak around zero.
- Ridge has a reasonable coefficient distribution without extra value like OLS has, nor many zeros as Lasso has.  

**3.3** The plots above show the overall distribution of coefficient values in each model, but do not show how each model treats individual coefficients. Build a plot which cleanly presents, for each feature in the data, 1) The coefficient assigned by OLS, 2) the coefficient assigned by ridge, and 3) the coefficient assigned by lasso.



```python
df = pd.DataFrame({'index': list(x_train.columns), 
                   'OLS': list(ols.coef_), 
                   'Ridge': list(ridge_reg.coef_), 
                   'Lasso': list(lasso_reg.coef_)})
df = df.set_index('index')
df.plot.barh(figsize=(8, 50), title='Model Coefficients');
```



![png](cs109a_hw4_web_files/cs109a_hw4_web_38_0.png)


**3.4** What trends do you see in the plot above? How do the three approaches handle the correlated pair `temp` and `atemp`?

We know that temp and atemp are highly correlated (HW3). The coefficients for OLS reflects multicollinearity. The magnitude of temp and atemp have been reduced considerably by Ridge and Lasso. Many of the variables have been reduced to zero for Lasso, where as Ridge has shrunk the coefficients. Storm seems to have no predictive power for any of the models. 

## Reflection 

**4.1** Reflect back on the `get_design_mats` function you built. Writing this function useful in your analysis? What issues might you have encountered if you copy/pasted the model-building code instead of tying it together in a function? Does a `get_design_mat` function seem wise in general, or are there better options?

get_design_mats() was very helpful and it encapsulates all the scaling, interaction terms and polynomial well. We were able to reuse the code instead of copy/paste and consequently increase chances of error prone code. Scikit learn has Pipelines, which may help us achieve similar/better results. 


**4.2** What are the costs and benefits of applying ridge/lasso regularization to an overfit OLS model, versus setting a specific degree of polynomial or forward selecting features for the model?

One major advantage of lasso over ridge is that it produces simpler and more interpretable models that involve only a subset of the predictors.  Lasso indirectly performs variable selection as it shrinks coefficients down to zero. 

Ridge regression’s advantage over least squares is rooted in the bias-variance trade-off. As $\lambda$ increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. Ridge regression works best in situations where the least squares estimates have high variance.

Ridge uses L2 penalty, whereas Lasso uses L1 penalty. 

Ridge regression also has substantial computational advantages over best subset selection, which requires searching through $2^p$ models.  Forward selection uses $1 + p(p + 1)/2$ models, where p is no. of predictors. In contrast, for any fixed value of $\lambda$, ridge regression only fits a single model, and the model-fitting procedure can be performed quite quickly. 

Higher degreee of polynomial will overfit to training data, resulting in poor fit to test data. 

Source: ISLR Book


**4.3** This pset posed a purely predictive goal: forecast ridership as accurately as possible. How important is interpretability in this context? Considering, e.g., your lasso and ridge models from Question 3, how would you react if the models predicted well, but the coefficient values didn't make sense once interpreted?

If forecasting ridership as accurately as possible is the ONLY goal, interpretability is not important in this context. 

Whether lasso or ridge, as long as it can predicte well, even if the coefficients don't make sense, I would still go with the the model. The obvious reason is that the model is working well and prediction accruracy is the only thing we care about, but it's also possible that the model is capturing some relationship that humans don't easily understand yet. 



**4.4** Reflect back on our original goal of helping BikeShare predict what demand will be like in the week ahead, and thus how many bikes they can bring in for maintenance. In your view, did we accomplish this goal? If yes, which model would you put into production and why? If not, which model came closest, what other analyses might you conduct, and how likely do you think they are to work

I think we have accomplished the goal reasonably well, with R-squared of the three models ranging from 35-39%.

I would recommend Lasso model, because (1) it's clearly better than an overfitted OLS model with higher R-squared on test set, and (2) it has similar R-squared but significantly fewer predictors than Ridge model, so that the BikeShare program management team can take explicit actions to raise revenue, attract more riders, and lower maintenance cost. 

