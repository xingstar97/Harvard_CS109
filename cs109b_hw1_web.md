---
title: title: H10
notebook: cs109b_hw1_web.ipynb
nav_include: 12
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2019-CS109B/master/content/styles/cs109.css").text
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
div.discussion {
	background-color: #ccffcc;
	border-color: #88E97A;
	border-left: 5px solid #0A8000; 
	padding: 0.5em;
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
## these two lines for JupyterHub only
import os
os.environ['R_HOME'] = "/usr/share/anaconda3/lib/R"

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.formula.api as sm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from sklearn.preprocessing import PolynomialFeatures

r_utils = importr('utils')

## if there are errors about missing R packages, uncomment and run the lines below:
## r_utils.install_packages('codetools')
## r_utils.install_packages('gam')
r_splines = importr('splines')
r_smooth_spline = robjects.r['smooth.spline'] #extract R function
r_gam_lib = importr('gam')
r_gam = r_gam_lib.gam #extract R function
r_glm = robjects.r['glm'] #extract R function
r_anova = robjects.r['anova'] #extract R function

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
```


## Modeling Seasonality of Airbnb Prices
In this problem, the task is to build a regression model to predict the price of an Airbnb rental for a given date. The data are provided in `calendar_train.csv` and `calendar_test.csv`, which contain availability and price data for Airbnb units in the Boston area from 2017 to 2018. Note that some of the rows in the `.csv` file refer to dates in the future. These refer to bookings that have been made far in advance.



```python
cal_train = pd.read_csv("data/calendar_train.csv")
cal_test = pd.read_csv("data/calendar_test.csv")
```




```python
cal_train.head()
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20872145</td>
      <td>9/21/18</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20872145</td>
      <td>9/19/18</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20872145</td>
      <td>9/18/18</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20872145</td>
      <td>9/17/18</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20872145</td>
      <td>9/16/18</td>
      <td>f</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>





```python
print(cal_train.shape, cal_test.shape)
```


    (734003, 4) (314572, 4)


### Exploratory Analysis

Visualize the average price by month and day of the week (i.e., Monday, Tuesday, etc.) for the training set. Point out any trends you notice and explain whether or not they make sense.

*Hint*: You will want to first convert the `date` column into Python dates using `datetime.datetime.strptime(arg1,arg2).date()` and providing the appropriate arguments.



```python
cal_train['date'] = pd.to_datetime(cal_train['date'], format="%m/%d/%y")
cal_test['date'] = pd.to_datetime(cal_test['date'], format="%m/%d/%y")
```




```python
cal_train['day'] = cal_train['date'].apply(lambda x: x.strftime('%a'))
cal_train['month'] = cal_train['date'].apply(lambda x: x.strftime('%m'))

cal_test['day'] = cal_test['date'].apply(lambda x: x.strftime('%a'))
cal_test['month'] = cal_test['date'].apply(lambda x: x.strftime('%m'))
```




```python
cal_train.head()
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>day</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20872145</td>
      <td>2018-09-21</td>
      <td>f</td>
      <td>NaN</td>
      <td>Fri</td>
      <td>09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20872145</td>
      <td>2018-09-19</td>
      <td>f</td>
      <td>NaN</td>
      <td>Wed</td>
      <td>09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20872145</td>
      <td>2018-09-18</td>
      <td>f</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20872145</td>
      <td>2018-09-17</td>
      <td>f</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20872145</td>
      <td>2018-09-16</td>
      <td>f</td>
      <td>NaN</td>
      <td>Sun</td>
      <td>09</td>
    </tr>
  </tbody>
</table>
</div>





```python
cal_test.head()
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
      <th>listing_id</th>
      <th>date</th>
      <th>available</th>
      <th>price</th>
      <th>day</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21205442</td>
      <td>2018-09-28</td>
      <td>t</td>
      <td>138.0</td>
      <td>Fri</td>
      <td>09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5166870</td>
      <td>2018-08-11</td>
      <td>t</td>
      <td>210.0</td>
      <td>Sat</td>
      <td>08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9698823</td>
      <td>2017-10-17</td>
      <td>f</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18894466</td>
      <td>2018-02-21</td>
      <td>f</td>
      <td>NaN</td>
      <td>Wed</td>
      <td>02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6765855</td>
      <td>2018-09-22</td>
      <td>f</td>
      <td>NaN</td>
      <td>Sat</td>
      <td>09</td>
    </tr>
  </tbody>
</table>
</div>





```python
cal_train = cal_train.loc[cal_train['available']=='t']
cal_test = cal_test.loc[cal_test['available']=='t']
```




```python
f, ax = plt.subplots(1, 2, figsize=(20,7))

g1 = sns.barplot(x='day', y='price', data=cal_train, order=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"], ax=ax[0])
g1.set_title('Average Price vs Day', fontsize=16)
g1.set_xlabel('Day', fontsize=16)
g1.set_ylabel('Price', fontsize=16)
g1.tick_params(labelsize=16)

g2 = sns.barplot(x='month', y='price', data=cal_train, ax=ax[1])
g2.set_title('Average Price vs Month', fontsize=16)
g2.set_xlabel('Month', fontsize=16)
g2.set_ylabel('Price', fontsize=16)
g2.tick_params(labelsize=16)

```



![png](cs109b_hw1_web_files/cs109b_hw1_web_12_0.png)


<div class="discussion"><b>Discussion: Trends</b></div>

Lower prices heading towards the middle of the week, with mimimal price on Wed, and higher prices heading towards the end of week, with highest prices on Friday and Saturday. This pattern is with rationale that people on average
may try to take time-off closer to the weekend.

The price stays low during winter time, from November to March, potentially attributed to the low demand of travel and cold weather in Boston. People normally take vacations in warm months, from late spring (starting in April) to early fall (ends in October). 


### Part 1a: Explore different regression models

Fit a regression model that uses the date as a predictor and predicts the average price of an Airbnb rental on that date. For this part of the question, you can ignore all other predictors besides the date. Fit the following models on the training set, and compare the $R^2$ of the fitted models on the test set. Include plots of the fitted models for each method.

*Hint*: You may want to convert the `date` column into a numerical variable by taking the difference in days between each date and the earliest date in the column.

1. Regression models with different basis functions:
    * Simple polynomials with degrees 5, 25, and 50
    * Cubic B-splines with the knots chosen by visual inspection of the data
    * Natural cubic splines with the degree of freedom chosen by cross-validation on the training set.
    
2. Smoothing spline model with the smoothness parameter chosen by cross-validation on the training set.

In each case, analyze the effect of the relevant tuning parameters on the training and test $R^2$ and give explanations for what you observe.

Is there a reason you would prefer one of these methods over the other (hint: you may want to consider $R^2$)?

*Hint*: The functions `bs` (for B-spline basis functions) and `ns` (for natural cubic spline basis functions) are available  in the `r_splines` library.




```python
cal_train['days_since'] = (cal_train.date - min(cal_train.date)).apply(lambda x: x.days)
cal_test['days_since'] = (cal_test.date - min(cal_train.date)).apply(lambda x: x.days)
```




```python
cal_train_agg = cal_train[['days_since', 'price']].groupby('days_since').mean().reset_index()
cal_test_agg = cal_test[['days_since', 'price']].groupby('days_since').mean().reset_index()
cal_train_agg.head()
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
      <th>days_since</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>370.173410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>409.298701</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>299.128713</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>282.768908</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>275.933202</td>
    </tr>
  </tbody>
</table>
</div>





```python
r_lm = robjects.r["lm"]
r_predict = robjects.r["predict"]
```




```python
r_price = robjects.FloatVector(cal_train_agg['price'])

r_days_since = robjects.FloatVector(cal_train_agg['days_since'])

cal_train_agg_r = robjects.DataFrame({"price": r_price, 
                                      "days_since": r_days_since})

```




```python
r_price = robjects.FloatVector(cal_test_agg['price'])

r_days_since = robjects.FloatVector(cal_test_agg['days_since'])

cal_test_agg_r = robjects.DataFrame({"price": r_price, 
                                     "days_since": r_days_since})

```




```python
poly5_formula = robjects.Formula("price ~ poly(days_since, degree=5, raw=TRUE)") 
model_cal_poly5 = r_lm(formula=poly5_formula, data=cal_train_agg_r) 

model_cal_poly5_r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_poly5, cal_train_agg_r))
model_cal_poly5_r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_poly5, cal_test_agg_r))
```




```python
poly25_formula = robjects.Formula("price ~ poly(days_since, degree=25, raw=TRUE)") 
model_cal_poly25 = r_lm(formula=poly25_formula, data=cal_train_agg_r) 

model_cal_poly25_r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_poly25, cal_train_agg_r))
model_cal_poly25_r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_poly25, cal_test_agg_r))
```




```python
poly50_formula = robjects.Formula("price ~ poly(days_since, degree=50, raw=TRUE)") 
model_cal_poly50 = r_lm(formula=poly50_formula, data=cal_train_agg_r) 

model_cal_poly50_r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_poly50, cal_train_agg_r))
model_cal_poly50_r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_poly50, cal_test_agg_r))
```




```python
print("Polynomial Degree: %d, R2 on training set: %.3f, R2 on test set: %.3f" 
      % (5, model_cal_poly5_r2_train, model_cal_poly5_r2_test))
print("Polynomial Degree: %d, R2 on training set: %.3f, R2 on test set: %.3f" 
      % (25, model_cal_poly25_r2_train, model_cal_poly25_r2_test))
print("Polynomial Degree: %d, R2 on training set: %.3f, R2 on test set: %.3f" 
      % (50, model_cal_poly50_r2_train, model_cal_poly50_r2_test))
```


    Polynomial Degree: 5, R2 on training set: 0.717, R2 on test set: 0.685
    Polynomial Degree: 25, R2 on training set: 0.763, R2 on test set: 0.726
    Polynomial Degree: 50, R2 on training set: 0.786, R2 on test set: 0.740




```python
f, ax = plt.subplots(1, 3, figsize=(20,7))

ax[0].scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax[0].plot(cal_test_agg['days_since'], r_predict(model_cal_poly5, cal_test_agg_r), color='red')
ax[0].set_title('Poly 5', fontsize=16)
ax[0].set_xlabel('Days Since', fontsize=16)
ax[0].set_ylabel('Average Price', fontsize=16)
ax[0].tick_params(labelsize=16)

ax[1].scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax[1].plot(cal_test_agg['days_since'], r_predict(model_cal_poly25, cal_test_agg_r), color='red')
ax[1].set_title('Poly 25', fontsize=16)
ax[1].set_xlabel('Days Since', fontsize=16)
ax[1].set_ylabel('Average Price', fontsize=16)
ax[1].tick_params(labelsize=16)

ax[2].scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax[2].plot(cal_test_agg['days_since'], r_predict(model_cal_poly50, cal_test_agg_r), color='red')
ax[2].set_title('Poly 50', fontsize=16)
ax[2].set_xlabel('Days Since', fontsize=16)
ax[2].set_ylabel('Average Price', fontsize=16)
ax[2].tick_params(labelsize=16)

```



![png](cs109b_hw1_web_files/cs109b_hw1_web_24_0.png)


**Discussion: Polynomial Fit**

As shown above, as the polynomial degree increases from 5 to 25 to 50, the r-squared on both training and test set increases. On the training set, the increase of r-squared is due to more degree of freedom. 

As seen in the plots, we risk overfitting the training set as we raise the polynomial degrees. The prediction line is becoming jumpy especially at polynomial degree of 50. 

Additionally, we have to set the raw=True in the poly function, otherwise we will get a runtime error of "'degree' must be less than number of unique points".



```python
r_splines = importr('splines')
```




```python
r_price = robjects.FloatVector(cal_train_agg['price'])

r_days_since = robjects.FloatVector(cal_train_agg['days_since'])

r_quarts = robjects.FloatVector(np.quantile(r_days_since, [.25,.5,.75])) 

bs_formula = robjects.Formula("price ~ bs(days_since, knots=r_quarts)")
bs_formula.environment['price'] = r_price
bs_formula.environment['days_since'] = r_days_since
bs_formula.environment['r_quarts'] = r_quarts

model_cal_bs = r_lm(formula=bs_formula)
```




```python
model_cal_bs_r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_bs, cal_train_agg_r))
model_cal_bs_r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_bs, cal_test_agg_r))

print("Cubic B−Splines, R2 on training set: %.3f, R2 on test set: %.3f" 
      % (model_cal_bs_r2_train, model_cal_bs_r2_test))
```


    Cubic B−Splines, R2 on training set: 0.725, R2 on test set: 0.693




```python
fig, ax = plt.subplots(1, 1, figsize=(11, 7))

ax.scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax.plot(cal_test_agg['days_since'], r_predict(model_cal_bs, cal_test_agg_r), color='red')
ax.set_title('Cubic B−Splines', fontsize=16)
ax.set_xlabel('Days Since', fontsize=16)
ax.set_ylabel('Average Price', fontsize=16)
ax.tick_params(labelsize=16)
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_29_0.png)


**Discussion: Cubic B−Splines**

As shown above, cubic b-splines yield a worse r-squared vs the polynomial fits. The performance of the fit is dependant on the knot selection by our visual inspection of the data. Thus, we can see that cubic s-splines are very sensitive to the knots placement choice. 



```python
r_price = robjects.FloatVector(cal_train_agg['price'])

r_days_since = robjects.FloatVector(cal_train_agg['days_since'])

list_df = [1, 5, 10, 15, 20, 25]
for df in list_df: 
    # Natural cubic spline formula
    ns_formula = robjects.Formula("price ~ ns(days_since, df=r_df)")
    ns_formula.environment['price'] = r_price
    ns_formula.environment['days_since'] = r_days_since
    ns_formula.environment['r_df'] = df
    
    # Fit natural cubic spline 
    model_cal_ns = r_lm(formula=ns_formula)

    # R-squared
    r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_ns, cal_train_agg_r))
    r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_ns, cal_test_agg_r))
    
    print("df: %d, R2 on training set: %.3f, R2 on test set: %.3f" % (df, r2_train, r2_test))

```


    df: 1, R2 on training set: 0.170, R2 on test set: 0.181
    df: 5, R2 on training set: 0.727, R2 on test set: 0.695
    df: 10, R2 on training set: 0.755, R2 on test set: 0.723
    df: 15, R2 on training set: 0.768, R2 on test set: 0.734
    df: 20, R2 on training set: 0.772, R2 on test set: 0.740
    df: 25, R2 on training set: 0.835, R2 on test set: 0.806




```python
def cv_degree(degree, df=cal_train_agg_r, n_splits=5):
    # CV score results
    results = []
    
    # Loop through folds
    for train_index, test_index in KFold(n_splits, shuffle=True).split(range(df.nrow)):
        train_index_r = robjects.IntVector(train_index)
        test_index_r = robjects.IntVector(test_index)
        
        sub_train_r = df.rx(train_index_r, True)
        sub_test_r = df.rx(test_index_r, True)

        # Natural cubic spline formula
        ns_formula = robjects.Formula("price ~ ns(days_since, df=r_degree)")
        ns_formula.environment['price'] = sub_train_r.rx2("price")
        ns_formula.environment['days_since'] = sub_train_r.rx2("days_since")
        ns_formula.environment['r_degree'] = degree

        # Fit nautral cubic spline
        model_cal_ns = r_lm(formula=ns_formula)
        
        # Validation score
        cv_score = r2_score(sub_test_r.rx2("price"), r_predict(model_cal_ns, sub_test_r))
        
        # Append the score
        results.append(cv_score)
    
    return np.mean(results)

```




```python
df_degree = pd.DataFrame({"degree": np.linspace(0,50,51), "cv_score": None})
df_degree['cv_score'] = df_degree['degree'].apply(lambda x: cv_degree(x))
```




```python
fig, ax = plt.subplots(1, 1, figsize=(11,7))
ax.plot(df_degree['degree'], df_degree['cv_score'], '-*')
ax.set_title('Cross-Validation R-Squared', fontsize=16)
ax.set_xlabel('Degree of Freedom', fontsize=16)
ax.set_ylabel('R-Squared', fontsize=16)
ax.tick_params(labelsize=16)
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_34_0.png)




```python
df_best_degree = df_degree.iloc[df_degree['cv_score'].idxmax()]
best_degree = df_best_degree['degree']
best_cv = df_best_degree['cv_score']
print("Best degree of freedom: %d" % best_degree)
```


    Best degree of freedom: 46




```python
r_price = robjects.FloatVector(cal_train_agg['price'])

r_days_since = robjects.FloatVector(cal_train_agg['days_since'])

ns_formula = robjects.Formula("price ~ ns(days_since, df=r_df)")
ns_formula.environment['price'] = r_price
ns_formula.environment['days_since'] = r_days_since
ns_formula.environment['r_df'] = best_degree

model_cal_ns = r_lm(formula=ns_formula)

r2_train = r2_score(cal_train_agg_r.rx2("price"), r_predict(model_cal_ns, cal_train_agg_r))
r2_test = r2_score(cal_test_agg_r.rx2("price"), r_predict(model_cal_ns, cal_test_agg_r))

print("Best degree of freedom: %d, R2 on training set: %.3f, R2 on test set: %.3f" % (best_degree, r2_train, r2_test))
```


    Best degree of freedom: 46, R2 on training set: 0.881, R2 on test set: 0.849




```python
fig, ax = plt.subplots(1, 1, figsize=(11, 7))

ax.scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax.plot(cal_test_agg['days_since'], r_predict(model_cal_ns, cal_test_agg_r), color='red')
ax.set_title('Natural Cubic Splines', fontsize=16)
ax.set_xlabel('Days Since', fontsize=16)
ax.set_ylabel('Average Price', fontsize=16)
ax.tick_params(labelsize=16)
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_37_0.png)


**Discussion: Natural cubic splines**

As shown above, natrual cubic-splines with cross-validated best degree of freedom yields the best r-squared on the test set. It beats both polynomial fits and cubic b-splines. Its best performance is due to the best parameter setting through cross validation. 
 



```python
r_smooth_spline = robjects.r['smooth.spline'] 
r_predict = robjects.r['predict'] 
```




```python
r_price = robjects.FloatVector(cal_train_agg['price'])

r_days_since = robjects.FloatVector(cal_train_agg['days_since'])

model_cal_spline = r_smooth_spline(r_days_since, r_price, cv=True) 

lambda_cv = float(str(model_cal_spline.rx2("lambda"))[4:-1])
print("The best Lambda: " + str(lambda_cv))
```


    The best Lambda: 1.904808e-08




```python
fig, ax = plt.subplots(1, 1, figsize=(11, 7))

ax.plot(np.array(tuple(model_cal_spline[0])), np.array(tuple(model_cal_spline[1])), color='red')
ax.scatter(x=cal_train_agg['days_since'], y=cal_train_agg['price'])
ax.set_title('Smoothing Splines', fontsize=16)
ax.set_xlabel('Days Since', fontsize=16)
ax.set_ylabel('Average Price', fontsize=16)
ax.tick_params(labelsize=16)
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_41_0.png)




```python
r2_train = r2_score(cal_train_agg_r.rx2("price"), np.array(tuple(model_cal_spline[1])))
r2_test = r2_score(cal_test_agg_r.rx2("price"), np.array(tuple(model_cal_spline[1])))

print("Smoothing Spline, R2 on training set: %.3f, R2 on test set: %.3f" % (r2_train, r2_test))
```


    Smoothing Spline, R2 on training set: 0.949, R2 on test set: 0.909


**Discussion: Smoothing Spline**

Smoothing Spline gives the highest r-squared on test set among all models.

**Discussion: Model Selection**

As shown above, the smoothing spline gives the highest r-squared on the test set among all models. We prefer smoothing splines over cubic b-splines, because picking the knots locations is very hard by visual inspection. We would also prefer smoothing splines over polynomial regression, because the parameter is optimal through cross-validation. 

However, the graph above shows that the smoothing spline is not really smooth. It is possible that the test set is pretty close to the training set, such that the r-squared on the test is the largest, but we should be very careful about overfitting here. 

### Part 1b: Adapting to weekends

Does the pattern of Airbnb pricing differ over the days of the week? Are the patterns on weekends different from those on weekdays? If so, we might benefit from using a different regression model for weekdays and weekends. Split the training and test data into two parts, one for weekdays and one for weekends, and fit a separate model for each training subset. Do the models yield a higher $R^2$ on the corresponding test subsets compared to the model fitted previously?



```python
cal_train_weekend = cal_train.loc[(cal_train['day'] == 'Fri') | (cal_train['day'] == 'Sat')]
cal_train_weekday = cal_train.loc[(cal_train['day'] != 'Fri') & (cal_train['day'] != 'Sat')]

cal_test_weekend = cal_test.loc[(cal_test['day'] == 'Fri') | (cal_test['day'] == 'Sat')]
cal_test_weekday = cal_test.loc[(cal_test['day'] != 'Fri') & (cal_test['day'] != 'Sat')]
```




```python
cal_train_weekend_agg = cal_train_weekend[['days_since', 'price']].groupby('days_since').mean().reset_index()
cal_train_weekday_agg = cal_train_weekday[['days_since', 'price']].groupby('days_since').mean().reset_index()

cal_test_weekend_agg = cal_test_weekend[['days_since', 'price']].groupby('days_since').mean().reset_index()
cal_test_weekday_agg = cal_test_weekday[['days_since', 'price']].groupby('days_since').mean().reset_index()
```




```python
r_lm = robjects.r["lm"]
r_predict = robjects.r["predict"]
```




```python
r_price = robjects.FloatVector(cal_train_weekend_agg['price'])
r_days_since = robjects.FloatVector(cal_train_weekend_agg['days_since'])
cal_train_weekend_agg_r = robjects.DataFrame({"price": r_price, 
                                              "days_since": r_days_since})

r_price = robjects.FloatVector(cal_train_weekday_agg['price'])
r_days_since = robjects.FloatVector(cal_train_weekday_agg['days_since'])
cal_train_weekday_agg_r = robjects.DataFrame({"price": r_price, 
                                              "days_since": r_days_since})
```




```python
r_price = robjects.FloatVector(cal_test_weekend_agg['price'])
r_days_since = robjects.FloatVector(cal_test_weekend_agg['days_since'])
cal_test_weekend_agg_r = robjects.DataFrame({"price": r_price, 
                                             "days_since": r_days_since})

r_price = robjects.FloatVector(cal_test_weekday_agg['price'])
r_days_since = robjects.FloatVector(cal_test_weekday_agg['days_since'])
cal_test_weekday_agg_r = robjects.DataFrame({"price": r_price, 
                                             "days_since": r_days_since})
```




```python
poly50_formula = robjects.Formula("price ~ poly(days_since, degree=50, raw=TRUE)") 

model_cal_poly50_weekend = r_lm(formula=poly50_formula, data=cal_train_weekend_agg_r) 
model_cal_poly50_weekend_pred = r_predict(model_cal_poly50_weekend, cal_test_weekend_agg_r)

model_cal_poly50_weekday = r_lm(formula=poly50_formula, data=cal_train_weekday_agg_r) 
model_cal_poly50_weekday_pred = r_predict(model_cal_poly50_weekday, cal_test_weekday_agg_r)
```




```python
model_cal_poly50_weekend_pred = list(np.array(model_cal_poly50_weekend_pred))
model_cal_poly50_weekday_pred = list(np.array(model_cal_poly50_weekday_pred))
model_cal_poly50_total_pred = model_cal_poly50_weekday_pred + model_cal_poly50_weekend_pred
```




```python
model_cal_poly50_weekend_real = list(np.array(cal_test_weekend_agg_r.rx2("price")))
model_cal_poly50_weekday_real = list(np.array(cal_test_weekday_agg_r.rx2("price")))
model_cal_poly50_total_real = model_cal_poly50_weekday_real + model_cal_poly50_weekend_real
```




```python
r2_total = r2_score(model_cal_poly50_total_real, model_cal_poly50_total_pred)
print("R2 on test set: %.3f" % (r2_total))
```


    R2 on test set: 0.782


**Discussion**

This approach does yeild an improved r-squared on the test set vs the best polynomial model.

### Part 1c: Going the Distance

You may have noticed from your scatterplots of average price versus day on the training set that there are a few days with abnormally high average prices.

Sort the training data in decreasing order of average price, extracting the 3 most expensive dates. Given what you know about Boston, how might you explain why these 3 days happen to be so expensive?



```python
cal_train.loc[cal_train['price'].idxmax()]

(pd.DataFrame(cal_train.groupby('date')['price'].mean()).sort_values(by=['price'],ascending=False))[:3]
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
      <th>price</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-04-14</th>
      <td>432.680761</td>
    </tr>
    <tr>
      <th>2018-04-16</th>
      <td>425.289528</td>
    </tr>
    <tr>
      <th>2018-04-15</th>
      <td>417.170404</td>
    </tr>
  </tbody>
</table>
</div>



**Discussion**

4/14/18 to 4/16/18 have the highest prices, which are the marathon weekends in Boston which caused the highest price.

## Predicting Airbnb Rental Price Through Listing Features

In this problem, we'll continue our exploration of Airbnb data by predicting price based on listing features. The data can be found in `listings_train.csv` and `listings_test.csv`.

First, visualize the relationship between each of the predictors and the response variable. Does it appear that some of the predictors have a nonlinear relationship with the response variable?



```python
listings_train = pd.read_csv("data/listings_train.csv")
listings_test = pd.read_csv("data/listings_test.csv")
```




```python
listings_train.head()
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
      <th>host_total_listings_count</th>
      <th>room_type</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>security_deposit</th>
      <th>cleaning_fee</th>
      <th>availability_365</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Private room</td>
      <td>42.347956</td>
      <td>-71.155196</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>52</td>
      <td>1</td>
      <td>65</td>
      <td>365</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>Entire home/apt</td>
      <td>42.349299</td>
      <td>-71.083470</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>110</td>
      <td>1</td>
      <td>104</td>
      <td>107</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>Entire home/apt</td>
      <td>42.341902</td>
      <td>-71.073792</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>67</td>
      <td>45</td>
      <td>56</td>
      <td>322</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Entire home/apt</td>
      <td>42.319235</td>
      <td>-71.105016</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
      <td>103</td>
      <td>8</td>
      <td>113</td>
      <td>341</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Entire home/apt</td>
      <td>42.346452</td>
      <td>-71.134896</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>24</td>
      <td>82</td>
      <td>41</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>





```python
print(listings_train.shape)
print(listings_test.shape)
```


    (4370, 12)
    (487, 12)




```python
features = [col for col in listings_train.columns if col != 'price']

f, axs = plt.subplots(6, 2, figsize = (15, 40))

for i in range(len(axs)):
    for j in range(len(axs[0])):
        index = i * len(axs[0]) + j
        if index != 11: 
            feature = features[index]
            axs[i][j].scatter(listings_train[feature], listings_train['price'])
            axs[i][j].set_title(feature, fontsize=14)
            axs[i][j].set_ylabel("Price", fontsize=14)
            
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_63_0.png)


**Discussion:**

It appears that some predictors don't have linear relationship with response variable (price).


### Part 2a: Polynomial Regression

Fit the following models on the training set and compare the $R^2$ score of the fitted models on the test set:
    
* Linear regression
* Regression with polynomial baseis functions of degree 3 (i.e., basis functions $x$, $x^2$, $x^3$ for each predictor $x$) for quantitative predictors.



```python
r_lm = robjects.r["lm"]
r_predict = robjects.r["predict"]
```




```python
listings_train = pd.get_dummies(listings_train, columns=['room_type'])
listings_train.drop(['room_type_Entire home/apt'], axis=1, inplace=True)

r_price = robjects.FloatVector(listings_train['price'])

r_host_total_listings_count = robjects.FloatVector(listings_train['host_total_listings_count'])
r_latitude = robjects.FloatVector(listings_train['latitude'])
r_longitude = robjects.FloatVector(listings_train['longitude'])
r_bathrooms = robjects.FloatVector(listings_train['bathrooms'])
r_bedrooms = robjects.FloatVector(listings_train['bedrooms'])
r_beds = robjects.FloatVector(listings_train['beds'])
r_security_deposit = robjects.FloatVector(listings_train['security_deposit'])
r_cleaning_fee = robjects.FloatVector(listings_train['cleaning_fee'])
r_availability_365 = robjects.FloatVector(listings_train['availability_365'])
r_number_of_reviews = robjects.FloatVector(listings_train['number_of_reviews'])
r_room_type_private_room = robjects.FloatVector(listings_train['room_type_Private room'])
r_room_type_shared_room = robjects.FloatVector(listings_train['room_type_Shared room'])

listings_train_r = robjects.DataFrame({"price": r_price, 
                                       "host_total_listings_count": r_host_total_listings_count, 
                                       "latitude": r_latitude, 
                                       "longitude": r_longitude, 
                                       "bathrooms": r_bathrooms, 
                                       "bedrooms": r_bedrooms, 
                                       "beds": r_beds, 
                                       "security_deposit": r_security_deposit, 
                                       "cleaning_fee": r_cleaning_fee, 
                                       "availability_365": r_availability_365, 
                                       "number_of_reviews": r_number_of_reviews, 
                                       "room_type_private_room": r_room_type_private_room, 
                                       "room_type_shared_room": r_room_type_shared_room
                                      })
```




```python
listings_test = pd.get_dummies(listings_test, columns=['room_type'])
listings_test.drop(['room_type_Entire home/apt'], axis=1, inplace=True)

r_price = robjects.FloatVector(listings_test['price'])

r_host_total_listings_count = robjects.FloatVector(listings_test['host_total_listings_count'])
r_latitude = robjects.FloatVector(listings_test['latitude'])
r_longitude = robjects.FloatVector(listings_test['longitude'])
r_bathrooms = robjects.FloatVector(listings_test['bathrooms'])
r_bedrooms = robjects.FloatVector(listings_test['bedrooms'])
r_beds = robjects.FloatVector(listings_test['beds'])
r_security_deposit = robjects.FloatVector(listings_test['security_deposit'])
r_cleaning_fee = robjects.FloatVector(listings_test['cleaning_fee'])
r_availability_365 = robjects.FloatVector(listings_test['availability_365'])
r_number_of_reviews = robjects.FloatVector(listings_test['number_of_reviews'])
r_room_type_private_room = robjects.FloatVector(listings_test['room_type_Private room'])
r_room_type_shared_room = robjects.FloatVector(listings_test['room_type_Shared room'])

listings_test_r = robjects.DataFrame({"price": r_price, 
                                       "host_total_listings_count": r_host_total_listings_count, 
                                       "latitude": r_latitude, 
                                       "longitude": r_longitude, 
                                       "bathrooms": r_bathrooms, 
                                       "bedrooms": r_bedrooms, 
                                       "beds": r_beds, 
                                       "security_deposit": r_security_deposit, 
                                       "cleaning_fee": r_cleaning_fee, 
                                       "availability_365": r_availability_365, 
                                       "number_of_reviews": r_number_of_reviews, 
                                       "room_type_private_room": r_room_type_private_room, 
                                       "room_type_shared_room": r_room_type_shared_room
                                      })
```




```python
listings_test_r
```






    <span>R/rpy2 DataFrame (487 x 13)</span>
    <table>
      <thead>
        <tr>
        
          <th>price</th>
        
          <th>host_total_listings_count</th>
        
          <th>latitude</th>
        
          <th>...</th>
        
          <th>number_of_reviews</th>
        
          <th>room_type_private_room</th>
        
          <th>room_type_shared_room</th>
        
        </tr>
      </thead>
      <tbody>
      
      <tr>
      
      <td>
        271.000000
      </td>
      
      <td>
        3.000000
      </td>
      
      <td>
        42.280279
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        17.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        290.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.285603
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        3.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        62.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.274408
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        3.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        55.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.287974
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        5.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        325.000000
      </td>
      
      <td>
        2.000000
      </td>
      
      <td>
        42.383755
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        250.000000
      </td>
      
      <td>
        9.000000
      </td>
      
      <td>
        42.383874
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        244.000000
      </td>
      
      <td>
        4.000000
      </td>
      
      <td>
        42.367770
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        147.000000
      </td>
      
      <td>
        4.000000
      </td>
      
      <td>
        42.365833
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      </tbody>
    </table>
    





```python
lm_formula = robjects.Formula("price ~ host_total_listings_count + latitude + longitude + bathrooms + bedrooms + beds + security_deposit + cleaning_fee  + availability_365 + number_of_reviews + room_type_private_room + room_type_shared_room")                
model_lm = r_lm(formula=lm_formula, data=listings_train_r)
```




```python
model_lm
```






    <span>ListVector with 12 elements.</span>
    <table>
      <tbody>
      
      <tr>
      <th>
        coefficients
      </th>
      <td>
        
    <span>FloatVector with 13 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -5377.615360
      </td>
      
      <td>
        -0.053929
      </td>
      
      <td>
        -69.000316
      </td>
      
      <td>
        -118.063740
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.031648
      </td>
      
      <td>
        -0.002727
      </td>
      
      <td>
        106.311617
      </td>
      
      <td>
        110.295113
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        residuals
      </th>
      <td>
        
    <span>FloatVector with 4370 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -193.399976
      </td>
      
      <td>
        8.592499
      </td>
      
      <td>
        -59.954208
      </td>
      
      <td>
        -59.902017
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -95.108145
      </td>
      
      <td>
        -155.837457
      </td>
      
      <td>
        -223.085566
      </td>
      
      <td>
        -63.828654
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        effects
      </th>
      <td>
        
    <span>FloatVector with 4370 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -11388.168680
      </td>
      
      <td>
        1396.863512
      </td>
      
      <td>
        1083.349525
      </td>
      
      <td>
        -577.841250
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -90.559934
      </td>
      
      <td>
        -151.442468
      </td>
      
      <td>
        -205.366093
      </td>
      
      <td>
        -61.207015
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        ...
      </th>
      <td>
        ...
      </td>
      </tr>
      
      <tr>
      <th>
        call
      </th>
      <td>
        
    <span>Vector with 3 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        SignatureTrans...
      </td>
      
      <td>
        Formula
      </td>
      
      <td>
        DataFrame
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        terms
      </th>
      <td>
        price ~ host_total_listings_count + latitude + longitude + bathrooms + 
    bedrooms + beds + security_deposit + cleaning_fee + availability_365 + 
    number_of_reviews + room_type_private_room + room_type_shared_room
attr(,"variables")
list(price, host_total_listings_count, latitude, longitude, bathrooms, 
    bedrooms, beds, security_deposit, cleaning_fee, availability_365, 
    number_of_reviews, room_type_private_room, room_type_shared_room)
attr(,"factors")
                          host_total_listings_count latitude longitude
price                                             0        0         0
host_total_listings_count                         1        0         0
latitude                                          0        1         0
longitude                                         0        0         1
bathrooms                                         0        0         0
bedrooms                                          0        0         0
beds                                              0        0         0
security_deposit                                  0        0         0
cleaning_fee                                      0        0         0
availability_365                                  0        0         0
number_of_reviews                                 0        0         0
room_type_private_room                            0        0         0
room_type_shared_room                             0        0         0
                          bathrooms bedrooms beds security_deposit cleaning_fee
price                             0        0    0                0            0
host_total_listings_count         0        0    0                0            0
latitude                          0        0    0                0            0
longitude                         0        0    0                0            0
bathrooms                         1        0    0                0            0
bedrooms                          0        1    0                0            0
beds                              0        0    1                0            0
security_deposit                  0        0    0                1            0
cleaning_fee                      0        0    0                0            1
availability_365                  0        0    0                0            0
number_of_reviews                 0        0    0                0            0
room_type_private_room            0        0    0                0            0
room_type_shared_room             0        0    0                0            0
                          availability_365 number_of_reviews
price                                    0                 0
host_total_listings_count                0                 0
latitude                                 0                 0
longitude                                0                 0
bathrooms                                0                 0
bedrooms                                 0                 0
beds                                     0                 0
security_deposit                         0                 0
cleaning_fee                             0                 0
availability_365                         1                 0
number_of_reviews                        0                 1
room_type_private_room                   0                 0
room_type_shared_room                    0                 0
                          room_type_private_room room_type_shared_room
price                                          0                     0
host_total_listings_count                      0                     0
latitude                                       0                     0
longitude                                      0                     0
bathrooms                                      0                     0
bedrooms                                       0                     0
beds                                           0                     0
security_deposit                               0                     0
cleaning_fee                                   0                     0
availability_365                               0                     0
number_of_reviews                              0                     0
room_type_private_room                         1                     0
room_type_shared_room                          0                     1
attr(,"term.labels")
 [1] "host_total_listings_count" "latitude"                 
 [3] "longitude"                 "bathrooms"                
 [5] "bedrooms"                  "beds"                     
 [7] "security_deposit"          "cleaning_fee"             
 [9] "availability_365"          "number_of_reviews"        
[11] "room_type_private_room"    "room_type_shared_room"    
attr(,"order")
 [1] 1 1 1 1 1 1 1 1 1 1 1 1
attr(,"intercept")
[1] 1
attr(,"response")
[1] 1
attr(,".Environment")
<environment: R_GlobalEnv>
attr(,"predvars")
list(price, host_total_listings_count, latitude, longitude, bathrooms, 
    bedrooms, beds, security_deposit, cleaning_fee, availability_365, 
    number_of_reviews, room_type_private_room, room_type_shared_room)
attr(,"dataClasses")
                    price host_total_listings_count                  latitude 
                "numeric"                 "numeric"                 "numeric" 
                longitude                 bathrooms                  bedrooms 
                "numeric"                 "numeric"                 "numeric" 
                     beds          security_deposit              cleaning_fee 
                "numeric"                 "numeric"                 "numeric" 
         availability_365         number_of_reviews    room_type_private_room 
                "numeric"                 "numeric"                 "numeric" 
    room_type_shared_room 
                "numeric" 

      </td>
      </tr>
      
      <tr>
      <th>
        model
      </th>
      <td>
        
    <span>R/rpy2 DataFrame (4370 x 13)</span>
    <table>
      <thead>
        <tr>
        
          <th>price</th>
        
          <th>host_total_listings_count</th>
        
          <th>latitude</th>
        
          <th>...</th>
        
          <th>number_of_reviews</th>
        
          <th>room_type_private_room</th>
        
          <th>room_type_shared_room</th>
        
        </tr>
      </thead>
      <tbody>
      
      <tr>
      
      <td>
        52.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.347956
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        26.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        110.000000
      </td>
      
      <td>
        85.000000
      </td>
      
      <td>
        42.349299
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        38.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        67.000000
      </td>
      
      <td>
        6.000000
      </td>
      
      <td>
        42.341902
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        9.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        103.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.319235
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        49.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        66.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.356422
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        2.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        78.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.346041
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        8.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.336550
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        12.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        61.000000
      </td>
      
      <td>
        13.000000
      </td>
      
      <td>
        42.358497
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        21.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      </tbody>
    </table>
    
      </td>
      </tr>
      
      </tbody>
    </table>
    





```python
lm_pred = r_predict(model_lm, listings_test_r)
```




```python
poly3_formula = robjects.Formula("price ~ poly(host_total_listings_count,3) + poly(latitude,3) + poly(longitude,3) + poly(bathrooms,3) + poly(bedrooms,3) + poly(beds,3) + poly(security_deposit,3) + poly(cleaning_fee,3) + poly(availability_365,3) + poly(number_of_reviews,3) + room_type_private_room + room_type_shared_room") 
model_poly3 = r_lm(formula=poly3_formula, data=listings_train_r) 
```




```python
model_poly3
```






    <span>ListVector with 12 elements.</span>
    <table>
      <tbody>
      
      <tr>
      <th>
        coefficients
      </th>
      <td>
        
    <span>FloatVector with 33 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        132.541978
      </td>
      
      <td>
        -748.699302
      </td>
      
      <td>
        -806.042742
      </td>
      
      <td>
        352.368994
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -84.439842
      </td>
      
      <td>
        -135.737312
      </td>
      
      <td>
        106.893731
      </td>
      
      <td>
        113.615668
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        residuals
      </th>
      <td>
        
    <span>FloatVector with 4370 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -185.932686
      </td>
      
      <td>
        -14.565577
      </td>
      
      <td>
        -56.013656
      </td>
      
      <td>
        -67.194555
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -88.197209
      </td>
      
      <td>
        -147.864859
      </td>
      
      <td>
        -226.379668
      </td>
      
      <td>
        -69.600381
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        effects
      </th>
      <td>
        
    <span>FloatVector with 4370 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -11388.168680
      </td>
      
      <td>
        1396.863512
      </td>
      
      <td>
        353.344736
      </td>
      
      <td>
        98.572511
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -87.674625
      </td>
      
      <td>
        -147.171309
      </td>
      
      <td>
        -222.546423
      </td>
      
      <td>
        -70.931824
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        ...
      </th>
      <td>
        ...
      </td>
      </tr>
      
      <tr>
      <th>
        call
      </th>
      <td>
        
    <span>Vector with 3 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        SignatureTrans...
      </td>
      
      <td>
        Formula
      </td>
      
      <td>
        DataFrame
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        terms
      </th>
      <td>
        price ~ poly(host_total_listings_count, 3) + poly(latitude, 3) + 
    poly(longitude, 3) + poly(bathrooms, 3) + poly(bedrooms, 
    3) + poly(beds, 3) + poly(security_deposit, 3) + poly(cleaning_fee, 
    3) + poly(availability_365, 3) + poly(number_of_reviews, 
    3) + room_type_private_room + room_type_shared_room
attr(,"variables")
list(price, poly(host_total_listings_count, 3), poly(latitude, 
    3), poly(longitude, 3), poly(bathrooms, 3), poly(bedrooms, 
    3), poly(beds, 3), poly(security_deposit, 3), poly(cleaning_fee, 
    3), poly(availability_365, 3), poly(number_of_reviews, 3), 
    room_type_private_room, room_type_shared_room)
attr(,"factors")
                                   poly(host_total_listings_count, 3)
price                                                               0
poly(host_total_listings_count, 3)                                  1
poly(latitude, 3)                                                   0
poly(longitude, 3)                                                  0
poly(bathrooms, 3)                                                  0
poly(bedrooms, 3)                                                   0
poly(beds, 3)                                                       0
poly(security_deposit, 3)                                           0
poly(cleaning_fee, 3)                                               0
poly(availability_365, 3)                                           0
poly(number_of_reviews, 3)                                          0
room_type_private_room                                              0
room_type_shared_room                                               0
                                   poly(latitude, 3) poly(longitude, 3)
price                                              0                  0
poly(host_total_listings_count, 3)                 0                  0
poly(latitude, 3)                                  1                  0
poly(longitude, 3)                                 0                  1
poly(bathrooms, 3)                                 0                  0
poly(bedrooms, 3)                                  0                  0
poly(beds, 3)                                      0                  0
poly(security_deposit, 3)                          0                  0
poly(cleaning_fee, 3)                              0                  0
poly(availability_365, 3)                          0                  0
poly(number_of_reviews, 3)                         0                  0
room_type_private_room                             0                  0
room_type_shared_room                              0                  0
                                   poly(bathrooms, 3) poly(bedrooms, 3)
price                                               0                 0
poly(host_total_listings_count, 3)                  0                 0
poly(latitude, 3)                                   0                 0
poly(longitude, 3)                                  0                 0
poly(bathrooms, 3)                                  1                 0
poly(bedrooms, 3)                                   0                 1
poly(beds, 3)                                       0                 0
poly(security_deposit, 3)                           0                 0
poly(cleaning_fee, 3)                               0                 0
poly(availability_365, 3)                           0                 0
poly(number_of_reviews, 3)                          0                 0
room_type_private_room                              0                 0
room_type_shared_room                               0                 0
                                   poly(beds, 3) poly(security_deposit, 3)
price                                          0                         0
poly(host_total_listings_count, 3)             0                         0
poly(latitude, 3)                              0                         0
poly(longitude, 3)                             0                         0
poly(bathrooms, 3)                             0                         0
poly(bedrooms, 3)                              0                         0
poly(beds, 3)                                  1                         0
poly(security_deposit, 3)                      0                         1
poly(cleaning_fee, 3)                          0                         0
poly(availability_365, 3)                      0                         0
poly(number_of_reviews, 3)                     0                         0
room_type_private_room                         0                         0
room_type_shared_room                          0                         0
                                   poly(cleaning_fee, 3)
price                                                  0
poly(host_total_listings_count, 3)                     0
poly(latitude, 3)                                      0
poly(longitude, 3)                                     0
poly(bathrooms, 3)                                     0
poly(bedrooms, 3)                                      0
poly(beds, 3)                                          0
poly(security_deposit, 3)                              0
poly(cleaning_fee, 3)                                  1
poly(availability_365, 3)                              0
poly(number_of_reviews, 3)                             0
room_type_private_room                                 0
room_type_shared_room                                  0
                                   poly(availability_365, 3)
price                                                      0
poly(host_total_listings_count, 3)                         0
poly(latitude, 3)                                          0
poly(longitude, 3)                                         0
poly(bathrooms, 3)                                         0
poly(bedrooms, 3)                                          0
poly(beds, 3)                                              0
poly(security_deposit, 3)                                  0
poly(cleaning_fee, 3)                                      0
poly(availability_365, 3)                                  1
poly(number_of_reviews, 3)                                 0
room_type_private_room                                     0
room_type_shared_room                                      0
                                   poly(number_of_reviews, 3)
price                                                       0
poly(host_total_listings_count, 3)                          0
poly(latitude, 3)                                           0
poly(longitude, 3)                                          0
poly(bathrooms, 3)                                          0
poly(bedrooms, 3)                                           0
poly(beds, 3)                                               0
poly(security_deposit, 3)                                   0
poly(cleaning_fee, 3)                                       0
poly(availability_365, 3)                                   0
poly(number_of_reviews, 3)                                  1
room_type_private_room                                      0
room_type_shared_room                                       0
                                   room_type_private_room room_type_shared_room
price                                                   0                     0
poly(host_total_listings_count, 3)                      0                     0
poly(latitude, 3)                                       0                     0
poly(longitude, 3)                                      0                     0
poly(bathrooms, 3)                                      0                     0
poly(bedrooms, 3)                                       0                     0
poly(beds, 3)                                           0                     0
poly(security_deposit, 3)                               0                     0
poly(cleaning_fee, 3)                                   0                     0
poly(availability_365, 3)                               0                     0
poly(number_of_reviews, 3)                              0                     0
room_type_private_room                                  1                     0
room_type_shared_room                                   0                     1
attr(,"term.labels")
 [1] "poly(host_total_listings_count, 3)" "poly(latitude, 3)"                 
 [3] "poly(longitude, 3)"                 "poly(bathrooms, 3)"                
 [5] "poly(bedrooms, 3)"                  "poly(beds, 3)"                     
 [7] "poly(security_deposit, 3)"          "poly(cleaning_fee, 3)"             
 [9] "poly(availability_365, 3)"          "poly(number_of_reviews, 3)"        
[11] "room_type_private_room"             "room_type_shared_room"             
attr(,"order")
 [1] 1 1 1 1 1 1 1 1 1 1 1 1
attr(,"intercept")
[1] 1
attr(,"response")
[1] 1
attr(,".Environment")
<environment: R_GlobalEnv>
attr(,"predvars")
list(price, poly(host_total_listings_count, 3, coefs = list(alpha = c(80.321052631579, 
772.353941769194, 578.561376670201), norm2 = c(1, 4370, 215361712.563158, 
11174544595087.9, 416628341321117248))), poly(latitude, 3, coefs = list(
    alpha = c(42.3399818359975, 42.3150745071687, 42.3176406361852
    ), norm2 = c(1, 4370, 2.78383658711709, 0.00358622714650651, 
    6.58344882613669e-06))), poly(longitude, 3, coefs = list(
    alpha = c(-71.0839519680804, -71.1010553318737, -71.0817622406347
    ), norm2 = c(1, 4370, 4.68517832448607, 0.00773682156178045, 
    1.84246400196555e-05))), poly(bathrooms, 3, coefs = list(
    alpha = c(1.24439359267735, 2.44931094588602, 3.77708135297255
    ), norm2 = c(1, 4370, 1122.98764302059, 1123.98882004749, 
    2595.5088959513))), poly(bedrooms, 3, coefs = list(alpha = c(1.34073226544622, 
2.77661464269462, 5.02625196649411), norm2 = c(1, 4370, 3315.64965675057, 
13105.9500714626, 150427.55706736))), poly(beds, 3, coefs = list(
    alpha = c(1.75400457665904, 6.17116523845605, 10.4498228982907
    ), norm2 = c(1, 4370, 6770.55491990847, 126980.121075494, 
    2327656.02865625))), poly(security_deposit, 3, coefs = list(
    alpha = c(12.0366132723112, 33.6975932100624, 30.1288005946447
    ), norm2 = c(1, 4370, 1203722.14187643, 174809863.501273, 
    28336684022.9624))), poly(cleaning_fee, 3, coefs = list(alpha = c(43.6389016018307, 
51.2685271563445, 70.4633622459553), norm2 = c(1, 4370, 6553058.18672769, 
4486449648.37124, 3381154384476.09))), poly(availability_365, 
    3, coefs = list(alpha = c(157.512356979405, 195.363123868919, 
    157.447112981862), norm2 = c(1, 4370, 83975505.8327231, 614994042172.714, 
    4608438292743009))), poly(number_of_reviews, 3, coefs = list(
    alpha = c(24.8425629290618, 178.881135892241, 223.720706563227
    ), norm2 = c(1, 4370, 8371717.68329519, 102943706789.36, 
    994871352139477))), room_type_private_room, room_type_shared_room)
attr(,"dataClasses")
                             price poly(host_total_listings_count, 3) 
                         "numeric"                        "nmatrix.3" 
                 poly(latitude, 3)                 poly(longitude, 3) 
                       "nmatrix.3"                        "nmatrix.3" 
                poly(bathrooms, 3)                  poly(bedrooms, 3) 
                       "nmatrix.3"                        "nmatrix.3" 
                     poly(beds, 3)          poly(security_deposit, 3) 
                       "nmatrix.3"                        "nmatrix.3" 
             poly(cleaning_fee, 3)          poly(availability_365, 3) 
                       "nmatrix.3"                        "nmatrix.3" 
        poly(number_of_reviews, 3)             room_type_private_room 
                       "nmatrix.3"                          "numeric" 
             room_type_shared_room 
                         "numeric" 

      </td>
      </tr>
      
      <tr>
      <th>
        model
      </th>
      <td>
        
    <span>R/rpy2 DataFrame (4370 x 13)</span>
    <table>
      <thead>
        <tr>
        
          <th>price</th>
        
          <th>poly(host_total_listings_count, 3)</th>
        
          <th>poly(latitude, 3)</th>
        
          <th>...</th>
        
          <th>poly(number_of_reviews, 3)</th>
        
          <th>room_type_private_room</th>
        
          <th>room_type_shared_room</th>
        
        </tr>
      </thead>
      <tbody>
      
      <tr>
      
      <td>
        52.000000
      </td>
      
      <td>
        -0.005405
      </td>
      
      <td>
        0.004779
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.000400
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        110.000000
      </td>
      
      <td>
        0.000319
      </td>
      
      <td>
        0.005584
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.004547
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        67.000000
      </td>
      
      <td>
        -0.005064
      </td>
      
      <td>
        0.001151
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -0.005475
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        103.000000
      </td>
      
      <td>
        -0.005405
      </td>
      
      <td>
        -0.012435
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.008349
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        66.000000
      </td>
      
      <td>
        -0.004274
      </td>
      
      <td>
        -0.007608
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -0.006030
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        78.000000
      </td>
      
      <td>
        -0.004274
      </td>
      
      <td>
        -0.008017
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -0.007125
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        8.000000
      </td>
      
      <td>
        -0.004274
      </td>
      
      <td>
        -0.003515
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.003480
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        61.000000
      </td>
      
      <td>
        0.003801
      </td>
      
      <td>
        -0.006638
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.009911
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      </tr>
      
      </tbody>
    </table>
    
      </td>
      </tr>
      
      </tbody>
    </table>
    





```python
poly3_pred = r_predict(model_poly3, listings_test_r)
```




```python
print("R-squared of linear regression on the test set: %.3f" % r2_score(listings_test_r.rx2("price"), lm_pred))
print("R-squared of polynomial regression on the test set: %.3f" % r2_score(listings_test_r.rx2("price"), poly3_pred))
```


    R-squared of linear regression on the test set: 0.185
    R-squared of polynomial regression on the test set: 0.239


**Discussion**

Adding the cubic terms increase R-squared on the test set compared to linear regression.

### Part 2b: Generalized Additive Model (GAM)

*Helpful Hint:  Please refer to the lecture ipynb for the code template to perform GAM*

Do you see any advantage in fitting an additive regression model to these data, compared to the above models?

1. Fit a GAM to the training set, and compare the test $R^2$ of the fitted model to the above models. You may use a smoothing spline basis function on each predictor, with the same smoothing parameter for each basis function, tuned using cross-validation on the training set.

2. Plot and examine the smooth of each predictor for the fitted GAM. What are some useful insights conveyed by these plots?




```python
r_gam_lib = importr('gam')
r_gam = r_gam_lib.gam
```




```python
formula_string = "price ~ s(host_total_listings_count, spar=1) + s(latitude, spar=1) + s(longitude, spar=1) + s(bathrooms, spar=1) + s(bedrooms, spar=1) + s(beds, spar=1) + s(security_deposit, spar=1) + s(cleaning_fee, spar=1) + s(availability_365, spar=1) + s(number_of_reviews, spar=1) + room_type_private_room + room_type_shared_room"

list_spar = [0, 0.25, 0.5, 0.75, 1]
for spar in list_spar: 
    spar_string = "spar=%f" % spar
    formula_spar = formula_string.replace("spar=1", spar_string)
    
    gam_formula = robjects.Formula(formula_spar)
    model_gam = r_gam(formula=gam_formula, data=listings_train_r, family="gaussian")

    gam_pred = r_predict(model_gam, listings_test_r)
    
    r2_train = r2_score(listings_train_r.rx2("price"), r_predict(model_gam, listings_train_r))
    r2_test = r2_score(listings_test_r.rx2("price"), r_predict(model_gam, listings_test_r))
    
    print("Spar: %.2f, R2 on training set: %.3f, R2 on test set: %.3f" % (spar, r2_train, r2_test))

```


    Spar: 0.00, R2 on training set: 0.405, R2 on test set: 0.165
    Spar: 0.25, R2 on training set: 0.375, R2 on test set: 0.222
    Spar: 0.50, R2 on training set: 0.318, R2 on test set: 0.243
    Spar: 0.75, R2 on training set: 0.285, R2 on test set: 0.238
    Spar: 1.00, R2 on training set: 0.272, R2 on test set: 0.226




```python
def cv_spar(spar, df=listings_train_r, n_splits=5):
    formula_string = "price ~ s(host_total_listings_count, spar=1) + s(latitude, spar=1) + s(longitude, spar=1) + s(bathrooms, spar=1) + s(bedrooms, spar=1) + s(beds, spar=1) + s(security_deposit, spar=1) + s(cleaning_fee, spar=1) + s(availability_365, spar=1) + s(number_of_reviews, spar=1) + room_type_private_room + room_type_shared_room"
    
    spar_string = "spar=%f" % spar
    formula_spar = formula_string.replace("spar=1", spar_string)
    gam_formula = robjects.Formula(formula_spar)
    
    # CV score results
    results = []
    
    # Loop through folds
    for train_index, test_index in KFold(n_splits).split(range(df.nrow)):
        train_index_r = robjects.IntVector(train_index)
        test_index_r = robjects.IntVector(test_index)
        
        sub_train_r = df.rx(train_index_r, True)
        sub_test_r = df.rx(test_index_r, True)

        # Fit GAM
        model_gam = r_gam(formula=gam_formula, data=sub_train_r, family="gaussian")
        cv_score = r2_score(sub_test_r.rx2("price"), r_predict(model_gam, sub_test_r))
        
        # Append the score
        results.append(cv_score)
    
    return np.mean(results)
    
    
```




```python
df_spar = pd.DataFrame({"spar": np.linspace(0,1,101), "cv_score": None})
df_spar['cv_score'] = df_spar['spar'].apply(lambda x: cv_spar(x))
```




```python
fig, ax = plt.subplots(1, 1, figsize=(11,7))
ax.plot(df_spar['spar'], df_spar['cv_score'], '-*')
ax.set_title('Cross-Validation R-Squared', fontsize=16)
ax.set_xlabel('Spar', fontsize=16)
ax.set_ylabel('R-Squared', fontsize=16)
ax.tick_params(labelsize=16)
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_83_0.png)




```python
df_best_spar = df_spar.iloc[df_spar['cv_score'].idxmax()]
best_spar = df_best_spar['spar']
best_cv = df_best_spar['cv_score']
print("Best spar: %.2f" % best_spar)
```


    Best spar: 0.98




```python
formula_string = "price ~ s(host_total_listings_count, spar=1) + s(latitude, spar=1) + s(longitude, spar=1) + s(bathrooms, spar=1) + s(bedrooms, spar=1) + s(beds, spar=1) + s(security_deposit, spar=1) + s(cleaning_fee, spar=1) + s(availability_365, spar=1) + s(number_of_reviews, spar=1) + room_type_private_room + room_type_shared_room"

spar_string = "spar=%f" % best_spar
formula_spar = formula_string.replace("spar=1", spar_string)
gam_formula = robjects.Formula(formula_spar)

model_gam = r_gam(formula=gam_formula, data=listings_train_r, family="gaussian")

gam_pred = r_predict(model_gam, listings_test_r)

r2_train = r2_score(listings_train_r.rx2("price"), r_predict(model_gam, listings_train_r))
r2_test = r2_score(listings_test_r.rx2("price"), r_predict(model_gam, listings_test_r))

print("Spar: %.2f, R2 on training set: %.3f, R2 on test set: %.3f" % (best_spar, r2_train, r2_test))
```


    Spar: 0.98, R2 on training set: 0.273, R2 on test set: 0.227


**Discussion**

GAM yields slightly lower test R2 compared to the previous regression models with polynomial
degree 3, but yields a more interpretable model.



```python
model_gam
```






    <span>ListVector with 35 elements.</span>
    <table>
      <tbody>
      
      <tr>
      <th>
        smooth.frame
      </th>
      <td>
        
    <span>R/rpy2 DataFrame (4370 x 10)</span>
    <table>
      <thead>
        <tr>
        
          <th>s(host_total_listings_count, spar = 0.98)</th>
        
          <th>s(latitude, spar = 0.98)</th>
        
          <th>s(longitude, spar = 0.98)</th>
        
          <th>...</th>
        
          <th>s(cleaning_fee, spar = 0.98)</th>
        
          <th>s(availability_365, spar = 0.98)</th>
        
          <th>s(number_of_reviews, spar = 0.98)</th>
        
        </tr>
      </thead>
      <tbody>
      
      <tr>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.347956
      </td>
      
      <td>
        -71.155196
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        65.000000
      </td>
      
      <td>
        365.000000
      </td>
      
      <td>
        26.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        85.000000
      </td>
      
      <td>
        42.349299
      </td>
      
      <td>
        -71.083470
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        104.000000
      </td>
      
      <td>
        107.000000
      </td>
      
      <td>
        38.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        6.000000
      </td>
      
      <td>
        42.341902
      </td>
      
      <td>
        -71.073792
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        56.000000
      </td>
      
      <td>
        322.000000
      </td>
      
      <td>
        9.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.319235
      </td>
      
      <td>
        -71.105016
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        113.000000
      </td>
      
      <td>
        341.000000
      </td>
      
      <td>
        49.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        ...
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.356422
      </td>
      
      <td>
        -71.059163
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        2.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.346041
      </td>
      
      <td>
        -71.073112
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        1.000000
      </td>
      
      <td>
        3.000000
      </td>
      
      <td>
        1.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        1.000000
      </td>
      
      <td>
        42.336550
      </td>
      
      <td>
        -71.086097
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        61.000000
      </td>
      
      <td>
        0.000000
      </td>
      
      <td>
        12.000000
      </td>
      
      </tr>
      
      <tr>
      
      <td>
        13.000000
      </td>
      
      <td>
        42.358497
      </td>
      
      <td>
        -71.062011
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        62.000000
      </td>
      
      <td>
        295.000000
      </td>
      
      <td>
        21.000000
      </td>
      
      </tr>
      
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        coefficients
      </th>
      <td>
        
    <span>FloatVector with 13 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -3556.755367
      </td>
      
      <td>
        -0.043795
      </td>
      
      <td>
        -138.003583
      </td>
      
      <td>
        -133.587954
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        0.015224
      </td>
      
      <td>
        0.019134
      </td>
      
      <td>
        107.832786
      </td>
      
      <td>
        115.593415
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        residuals
      </th>
      <td>
        
    <span>FloatVector with 4370 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        -190.973756
      </td>
      
      <td>
        -9.252273
      </td>
      
      <td>
        -57.328102
      </td>
      
      <td>
        -53.439431
      </td>
      
      <td>
        ...
      </td>
      
      <td>
        -83.930034
      </td>
      
      <td>
        -147.497416
      </td>
      
      <td>
        -225.963934
      </td>
      
      <td>
        -66.203368
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        ...
      </th>
      <td>
        ...
      </td>
      </tr>
      
      <tr>
      <th>
        method
      </th>
      <td>
        
    <span>StrVector with 1 elements.</span>
    <table>
      <tbody>
      <tr>
      
      <td>
        'glm.fit'
      </td>
      
      </tr>
      </tbody>
    </table>
    
      </td>
      </tr>
      
      <tr>
      <th>
        contrasts
      </th>
      <td>
        NULL
      </td>
      </tr>
      
      <tr>
      <th>
        xlevels
      </th>
      <td>
        
    <span>ListVector with 0 elements.</span>
    <table>
      <tbody>
      
      </tbody>
    </table>
    
      </td>
      </tr>
      
      </tbody>
    </table>
    





```python
%load_ext rpy2.ipython
```




```python
%R -i model_gam plot(model_gam, se=TRUE, scale=25);
```



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_0.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_1.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_2.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_3.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_4.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_5.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_6.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_7.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_8.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_9.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_10.png)



![png](cs109b_hw1_web_files/cs109b_hw1_web_89_11.png)


**Discussion**

GAM capturs the nonlinearities of some variables, for example longitude, number_of_reviews, availability_365, security_deposit. However, some varialbes appear to have linear relationships with the response variable, for example bathrooms, beds, bedrooms. 


### Part 2c: Putting it All Together

Based on your analysis for problems 1 and 2, what advice would you give a frugal visitor to Boston looking to save some money on an Airbnb rental?


**Discussion**

Based on problem 1, I would advice visitors to visit Boston during the weekdays and winter months from Novemenber to March to save money. Also, I would advice visitors to avoid the days around Marathon Monday if they are not runners. 

Based on problem 2, it really saves money by choosing the rooms with low number of rooms. More bedrooms seem to be associated with higher cost more than number of beds, so it saves money by sharing beds in the same room. Choosing the rooms with high number of reviews is also a way to save money, so looking for the highest reviewed rental listings. As a last advice, I would advice to choose the hostings with large total number of listings. They may have the pressue to rent the listings out to minimize operating cost. 


