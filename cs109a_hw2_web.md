---
title: H2
notebook: cs109a_hw2_web.ipynb
nav_include: 4
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
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.api import OLS
%matplotlib inline
```


## Predicting Taxi Pickups in NYC

In this homework, we will explore k-nearest neighbor and linear regression methods for predicting a quantitative variable. **Specifically, we will build regression models that can predict the number of taxi pickups in New York city at any given time of the day.** These prediction models will be useful, for example, in monitoring traffic in the city.

**1.1**. Use pandas to load the dataset from the csv file `dataset_1.csv` into a pandas data frame.  Use the `train_test_split` method from `sklearn` with a `random_state` of 42 and a `test_size` of 0.2 to split the dataset into training and test sets.  Store your train set dataframe in the variable `train_data`.  Store your test set dataframe in the variable `test_data`.



```python
data = pd.read_csv("data/dataset_1.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
```


**1.2**. Generate a scatter plot of the training data points with clear labels on the x and y axes. The time of the day on the x-axis and the number of taxi pickups on the y-axis.  Make sure to title your plot.



```python
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.grid(True, lw=1.75, ls='--', alpha=0.15)
ax.scatter(train_data['TimeMin'], train_data['PickupCount'], c='b', alpha=0.5, label='Training Set')
ax.set_title(r'Training Set - Taxi Pickups in NYC')
ax.set_xlabel(r'Time of the Day (Min)')
ax.set_ylabel(r'Number of Taxi Pickups')
ax.legend(loc='upper right');
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_7_0.png)


**1.3**. Does the pattern of taxi pickups make intuitive sense to you? 

**Answer:** The pattern of pickups seems to bear out the social patterns you'd expect in a major urban metropolis like New York.  We see instances of very high pickup counts between midnight and 5 a.m, when people take cabs home as bars close (in a city that never sleeps as opposed to a quiet academic town like Boston). Then you see a linear trend of pickups starting at a low point in the early morning (just after 5 a.m.) during the beginning of the morning commute when you'd expect very little social going on and steadily increasing to the common social hours in the evening at night when you'd expect people to congregate for dinner, shows, concerts, etc. There does appear to be a mid-morning surge around 8am to 10:30am, perhaps as some people travel to work via taxi.  


## k-Nearest Neighbors

**2.1**. Choose `TimeMin` as your feature variable and `PickupCount` as your response variable.  Create a dictionary of `KNeighborsRegressor` objects and call it `KNNModels`.  Let the key for your `KNNmodels` dictionary be the value of $k$ and the value be the corresponding `KNeighborsRegressor` object. For $k \in \{1, 10, 75, 250, 500, 750, 1000\}$, fit k-NN regressor models on the training set (`train_data`). 



```python
KNNModels = {}
K = [1,10,75,250,500,750,1000]

for k in K:
    KNNModels[k] = KNeighborsRegressor(n_neighbors=k)
    KNNModels[k].fit(train_data[['TimeMin']], train_data[['PickupCount']])
```


**2.2**.  For each $k$ on the training set, overlay a scatter plot of the actual values of `PickupCount` vs. `TimeMin` with a scatter plot of **predictions** for `PickupCount` vs  `TimeMin`.  Do the same for the test set.  You should have one figure with 2 x 7 total subplots; for each $k$ the figure should have two subplots, one subplot for the training set and one for the test set. 




```python
f, axarr = plt.subplots(2, 7, figsize=(15, 6), sharex=True, sharey=True)

for i, k in enumerate(K):     
    predicted_train = KNNModels[k].predict(train_data[['TimeMin']])
    axarr[0, i].scatter(train_data['TimeMin'], train_data['PickupCount'],marker='*',label='actual')
    axarr[0, i].scatter(train_data['TimeMin'], predicted_train,marker='^',label='prediction')
    axarr[0, i].set_title('Training set k=%i'%k)
    axarr[0, i].set_xlabel('Time(Min)')
    axarr[0, i].set_ylabel('Number of Pickups')
    axarr[0, i].legend(prop={'size': 8})
    
    predicted_test = KNNModels[k].predict(test_data[['TimeMin']])
    axarr[1, i].scatter(test_data['TimeMin'], test_data['PickupCount'],marker='*',label='actual')
    axarr[1, i].scatter(test_data['TimeMin'], predicted_test,marker='^',label='prediction')
    axarr[1, i].set_title('Testing set k=%i'%k)
    axarr[1, i].set_ylabel('Number of Pickups')
    axarr[1, i].set_xlabel('Time(Min)')
    axarr[1, i].legend(prop={'size': 8})

f.suptitle('Number of taxi pickups during a day',fontsize=14)
f.tight_layout()
f.subplots_adjust(top=0.85)
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_13_0.png)


**2.3**. Report the $R^2$ score for the fitted models on both the training and test sets for each $k$ (reporting the values in tabular form is encouraged).



```python
r2_train = []
r2_test = []
K = [1,10,75,250,500,750,1000]

for k in K:
    r2_train.append(KNNModels[k].score(train_data[['TimeMin']],train_data[['PickupCount']]))
    r2_test.append(KNNModels[k].score(test_data[['TimeMin']],test_data[['PickupCount']]))
    
r2 = pd.DataFrame(np.transpose([K,r2_train,r2_test]), columns=['K', 'R2_Train', 'R2_Test'])
```


**2.4**. Plot, in a single figure, the $R^2$ values from the model on the training and test set as a function of $k$.




```python
plt.plot(r2['K'], r2['R2_Train'], '*-', label='Train')
plt.plot(r2['K'], r2['R2_Test'], '^-', label='Test')
plt.legend()
plt.title('R2 for Train and Test Dataset')
plt.xlabel('K')
plt.ylabel('R2')
plt.show()
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_17_0.png)


**2.5 Discuss the results**

1. If $n$ is the number of observations in the training set, what can you say about a k-NN regression model that uses $k = n$?
   
    A k-NN regression model that used $k = n$ is the equivalent of using the mean of the response variable values for all the points of the dataset as a prediction model.


2. What does an $R^2$ score of $0$ mean?

    An $R^2$ value of 0 indicates a model making predictions equivalent to a model using a constant prediction of the data's mean (and as such explains none of the variation around the mean). In k-NN Regression, an example would be the model with $k = n$ or in this case $k = 1000$
  
  
3. What would a negative $R^2$ score mean?  Are any of the calculated $R^2$ you observe negative?

    None of the calculated $R^2$ values in this case on the training set are negative.  We see negative $R^2$ values for $k = 1$ and $k = 1000$ on the test set (although the test set $R^2$ value for $k = 1000$ is very close to 0).  A negative $R^2$ value indicate a model making predictions less accurate than using a constant prediction (for any configuration of features) of the mean of all response variable values.  Our observations of a highly negative $R^2$ score for $k = 1$ on the test set means that predictive value of the 1-NN model is very poor and 1-NN would be a worse model for our data than just taking the average value.  For $k = 1000$ the difference between the observed $R^2$ score on the test set and 0 is due to stochasticity and 1000-NN has a predictive power essentially equivalent to taking the average value on the training set as a prediction (in this particular case it so happens that 1000-NN is exactly the same model as using the average value of the training set for a prediction). 
    
    
4. Do the training and test $R^2$ plots exhibit different trends?  Describe.

    The training and test plots of $R^2$ exhibit different trends, as for small $k$, the model overfits the data, so it achieves a very good $R^2$ on the training set and a very poor $R^2$ on the test data. At large $k$ values the model underfits. Although it performs equally well on the train and test data, it's not doing as well on either one as it did at a different value of $k$.
  
  
5. How does the value of $k$ affect the fitted model and in particular the training and test $R^2$ values?

    The lower the value of $k$, the more variance in the predictions. The higher the value of $k$, the smoother the prediction. On the test set, greater $k$ decreases overfitting, but too large of $k$ does not allow for enough variation for an accurate prediction, so the test $R^2$ increases to a point and then decreases. Because the training $R^2$ benefits from the overfitting, the training $R^2$ only decreases as $k$ increases.  From an inspection of the plot, an ideal value of $k$ looks to be around 75.
  
  
6. What is the best value of $k$ and what are the corresponding training/test set $R^2$ values?

    Based on test set $R^2$ scores, the best value of $k$ is 75 with a training set $R^2$ score of 0.445 and a test set score of 0.390. Note that *best* refers to performance on the test set, the set on which the model can be evaluated.

## Simple Linear Regression

We next consider simple linear regression, which we know from lecture is a parametric approach for regression that assumes that the response variable has a linear relationship with the predictor.  Use the `statsmodels` module for Linear Regression. This module has built-in functions to summarize the results of regression and to compute confidence intervals for estimated regression parameters.  

**3.1**. Again choose `TimeMin` as your predictor and `PickupCount` as your response variable.  Create a `OLS` class instance and use it to fit a Linear Regression model on the training set (`train_data`).  Store your fitted model in the variable `OLSModel`.



```python
x_train, y_train = train_data['TimeMin'], train_data['PickupCount']
x_test, y_test = test_data['TimeMin'], test_data['PickupCount']

## create the X matrix by appending a column of ones to x_train
X_train = sm.add_constant(x_train)

## build the OLS model from the training data
OLSModel = sm.OLS(y_train, X_train)

## fit and save regression info 
texi_ols = OLSModel.fit()

print(texi_ols.summary())
```


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:            PickupCount   R-squared:                       0.243
    Model:                            OLS   Adj. R-squared:                  0.242
    Method:                 Least Squares   F-statistic:                     320.4
    Date:                Sun, 08 Sep 2019   Prob (F-statistic):           2.34e-62
    Time:                        09:54:44   Log-Likelihood:                -4232.9
    No. Observations:                1000   AIC:                             8470.
    Df Residuals:                     998   BIC:                             8480.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         16.7506      1.058     15.838      0.000      14.675      18.826
    TimeMin        0.0233      0.001     17.900      0.000       0.021       0.026
    ==============================================================================
    Omnibus:                      203.688   Durbin-Watson:                   1.910
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              462.910
    Skew:                           1.111   Prob(JB):                    3.02e-101
    Kurtosis:                       5.485   Cond. No.                     1.63e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.63e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.


**3.2**. Re-create your plot from 2.2 using the predictions from `OLSModel` on the training and test set. You should have one figure with two subplots, one subplot for the training set and one for the test set.



```python
fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].grid(True, lw=1.75, ls='--', alpha=0.15)
ax[0].scatter(x_train, y_train, c='b', alpha=0.5, label='Training Set')
ax[0].plot(x_train, texi_ols.fittedvalues, c='r', alpha=0.5, label='OLS Regression Line')
ax[0].set_title(r'Training Set')
ax[0].set_xlabel(r'Time of the Day (Min)')
ax[0].set_ylabel(r'Number of Taxi Pickups')
ax[0].legend(loc='upper right')

ax[1].grid(True, lw=1.75, ls='--', alpha=0.15)
ax[1].scatter(x_test, y_test, c='g', alpha=0.5, label='Testing Set')
ax[1].plot(x_train, texi_ols.fittedvalues, c='r', alpha=0.5, label='OLS Regression Line')
ax[1].set_title(r'Testing Set')
ax[1].set_xlabel(r'Time of the Day (Min)')
ax[1].set_ylabel(r'Number of Taxi Pickups')
ax[1].legend(loc='upper right')

fig.suptitle('Numer of taxi pickups during a day', fontsize=14);
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_24_0.png)


**3.3**. Report the $R^2$ score for the fitted model on both the training and test sets.




```python
X_test = sm.add_constant(x_test)
print("R-squared for training set: {:.3f}".format(texi_ols.rsquared))
print("R-squared for testing set: {:.3f}".format(r2_score(y_test, texi_ols.predict(X_test))))
```


    R-squared for training set: 0.243
    R-squared for testing set: 0.241


**3.4**. Report the slope and intercept values for the fitted linear model.  




```python
beta0 = texi_ols.params[0]
beta1 = texi_ols.params[1]
print("Intercept of the OLS: {:.2f}".format(beta0))
print("Slope of the OLS: {:.2f}".format(beta1))
```


    Intercept of the OLS: 16.75
    Slope of the OLS: 0.02


**3.5**. Report the $95\%$ confidence interval for the slope and intercept.




```python
thresh = 0.05
intervals = texi_ols.conf_int(alpha=thresh)
intervals = intervals.rename(index=str, columns={0:str(thresh/2*100)+"%",1:str((1-thresh/2)*100)+"%"})
intervals
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
      <th>2.5%</th>
      <th>97.5%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>const</th>
      <td>14.675141</td>
      <td>18.826062</td>
    </tr>
    <tr>
      <th>TimeMin</th>
      <td>0.020777</td>
      <td>0.025893</td>
    </tr>
  </tbody>
</table>
</div>



**3.6**. Create a scatter plot of the residuals ($e = y - \hat{y}$) of the linear regression model on the training set as a function of the predictor variable (i.e. `TimeMin`). Place on your plot a horizontal line denoting the constant zero residual.  




```python
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.grid(True, lw=1.75, ls='--', alpha=0.15)
ax.scatter(x_train, texi_ols.resid, c='b', alpha=0.7, label='Residuals')
ax.plot(x_train, [0] * len(x_train), c='r', alpha=0.8, label='Constant Zero Residual')
ax.set_title(r'Training Set - Regression residuals')
ax.set_xlabel(r'Time of the Day (Min)')
ax.set_ylabel(r'Residuals')
ax.legend(loc='upper right')
plt.show()
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_32_0.png)


**3.7 Discuss the results:**

1. How does the test $R^2$ score compare with the best test $R^2$ value obtained with k-NN regression?
  
  The test $R^2$ is lower for Linear Regression than for k-NN regression for all but the most suboptimal values of $k$ ($k \approx 0$ or $k \approx n$).  This isn't surprising since there are various indicators that a linear regression model isn't an ideal model for this particular choice of data and feature space.
  
  
2. What does the sign of the slope of the fitted linear model convey about the data?
  
  The positive slope implies that the number of pickups increases throughout the day. The slops is positive for all values within the confidence interval.
  
  
3. Based on the $95\%$ confidence interval, do you consider the estimates of the model parameters to be reliable?
  
  The estimates for slope and intercept are reasonably precise. The intercept is estimated to fall between around 14 to 18 on data that ranges from 0-100, which reasonably small though certainly far from perfect. The slope, it seems, is very precise, estimated to be between .020 and .025. In practical terms, using the lower end would predict 29 pickups (plus the intercept) at 11:59pm and using the upper bounds would predict 36 pickups (plus the intercept) at 11:59 pm, which is a fairly tight range. Our uncertainty in the value of the slope is small enough to only moderately impact our overall uncertainty, even at the extremes of the data.
  
  
4. Do you expect a $99\%$ confidence interval for the slope and intercept to be tighter (shorter) or looser (longer) than the $95\%$ confidence intervals? Briefly explain your answer.
  
  We'd expect a 99% confidence interval to be looser, as it should allow for an at least even wider possibility of values that are believable, or consistent with the data. With increased confidence level, even more values become plausible so the interval is lengthened on both sides. <br>
  
  
5. Based on the residuals plot that you made, discuss whether or not the assumption of linearity is valid for this data.
  
  The assumption of linearity does not seem to be perfectly justified, as the residuals are not scattered randomly around 0 and there is a clear structure! <br>
<br>

6. Based on the data structure, what restriction on the model would you put at the endpoints (at $x\approx0$ and $x\approx1440$)?   What does this say about the linearity assumption?

    Looking at x=0 and x=1440, y values should be same because it’s only a  minute difference in time. That’s not the case for y though. This invalidates the linearity assumption. This can be checked with the residual plots.

  

## Outliers 

**4.1**. We've provided you with two files `outliers_train.txt` and `outliers_test.txt` corresponding to training set and test set data.  What does a visual inspection of training set tell you about the existence of outliers in the data?
 



```python
outliers_train = pd.read_csv("data/outliers_train.csv")
outliers_test = pd.read_csv("data/outliers_test.csv")

def scatterplot(x, y, title):
    fig, ax = plt.subplots(1, 1)
    ax.grid(True, lw=1.75, ls='--', alpha=0.15)
    ax.scatter(x_train, y_train)
    ax.set_title(title)
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    
    return ax

x_train, y_train = outliers_train['X'], outliers_train['Y']
x_test, y_test = outliers_test['X'], outliers_test['Y']
scatterplot(x_train, y_train, 'Scatter Plot of Y vs X');
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_37_0.png)


Visual inspection of the training set tells me that some potential outliers are: 

- Top Left Cornor: two points around (-2.0, 300)
- Bottom Right Cornor: one point around (2.0, -300)

**4.2**. Choose `X` as your feature variable and `Y` as your response variable.  Use `statsmodel` to create a Linear Regression model on the training set data.  Store your model in the variable `OutlierOLSModel`.



```python
X_train = sm.add_constant(x_train)
OutlierOLSModel = sm.OLS(y_train, X_train)
results_sm = OutlierOLSModel.fit()
print(results_sm.summary())
```


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      Y   R-squared:                       0.084
    Model:                            OLS   Adj. R-squared:                  0.066
    Method:                 Least Squares   F-statistic:                     4.689
    Date:                Sun, 08 Sep 2019   Prob (F-statistic):             0.0351
    Time:                        09:54:45   Log-Likelihood:                -343.59
    No. Observations:                  53   AIC:                             691.2
    Df Residuals:                      51   BIC:                             695.1
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -9.5063     22.192     -0.428      0.670     -54.059      35.046
    X             47.3554     21.869      2.165      0.035       3.452      91.259
    ==============================================================================
    Omnibus:                        2.102   Durbin-Watson:                   1.758
    Prob(Omnibus):                  0.350   Jarque-Bera (JB):                1.251
    Skew:                           0.215   Prob(JB):                        0.535
    Kurtosis:                       3.617   Cond. No.                         1.06
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


**4.3**. You're given the knowledge ahead of time that there are 3 outliers in the training set data.  The test set data doesn't have any outliers.  You want to remove the 3 outliers in order to get the optimal intercept and slope.  In the case that you're sure of the existence and number (3) of outliers ahead of time, one potential brute force method to outlier detection might be to find the best Linear Regression model on all possible subsets of the training set data with 3 points removed.  Using this method, how many times will you have to calculate the Linear Regression coefficients on the training data?  

Picking 3 out of 53 samples gives in total (53 x 52 x 51) / (3 x 2) = 23,426 combinations.

**4.4**  In CS109 we're strong believers that creating heuristic models is a great way to build intuition. In that spirit, construct an approximate algorithm to find the 3 outlier candidates in the training data by taking advantage of the Linear Regression residuals. Place your algorithm in the function `find_outliers_simple`.  It should take the parameters `dataset_x` and `dataset_y` representing your features and response variable values (make sure your response variable is stored as a numpy column vector).  The return value should be a list `outlier_indices` representing the indices of the 3 outliers in the original datasets you passed in.  Remove the outliers that your algorithm identified, use `statsmodels` to create a Linear Regression model on the remaining training set data, and store your model in the variable `OutlierFreeSimpleModel`.



```python
def find_outliers_simple(dataset_x, dataset_y):
    X_train = sm.add_constant(dataset_x)
    OutlierOLSModel = sm.OLS(dataset_y, X_train)
    results_sm = OutlierOLSModel.fit()
    
    ## create a dataframe for labeling outliers
    df = pd.DataFrame({'X': dataset_x, 
                       'Y': dataset_y, 
                       'resid': results_sm.resid, 
                       'resid_abs': abs(results_sm.resid)
                      })
    
    df.sort_values(by=['resid_abs'], ascending=False, inplace=True)    
    outlier_indices = list(df.iloc[:3].index)
    
    return outlier_indices

## get outliers
outlier_indices = find_outliers_simple(x_train, y_train)
outliers_train_clean = outliers_train.drop(outlier_indices)

## calculate outlier model
x_train_clean, y_train_clean = outliers_train_clean['X'], outliers_train_clean['Y']
X_train_clean = sm.add_constant(x_train_clean)
OutlierFreeSimpleModel = sm.OLS(y_train_clean, X_train_clean)
results_sm_clean = OutlierFreeSimpleModel.fit()
print(results_sm_clean.summary())
```


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      Y   R-squared:                       0.404
    Model:                            OLS   Adj. R-squared:                  0.391
    Method:                 Least Squares   F-statistic:                     32.50
    Date:                Sun, 08 Sep 2019   Prob (F-statistic):           7.16e-07
    Time:                        09:54:45   Log-Likelihood:                -309.21
    No. Observations:                  50   AIC:                             622.4
    Df Residuals:                      48   BIC:                             626.2
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const        -17.4796     16.944     -1.032      0.307     -51.547      16.588
    X            104.8467     18.392      5.701      0.000      67.867     141.827
    ==============================================================================
    Omnibus:                        0.600   Durbin-Watson:                   1.683
    Prob(Omnibus):                  0.741   Jarque-Bera (JB):                0.673
    Skew:                          -0.238   Prob(JB):                        0.714
    Kurtosis:                       2.689   Cond. No.                         1.09
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


**4.5** Create a figure with two subplots: the first is a scatterplot where the color of the points denotes the outliers from the non-outliers in the training set, and include two regression lines on this scatterplot: one fitted with the outliers included and one fitted with the outlier removed (all on the training set).  The second plot should include a scatterplot of points from the test set with the same two regression lines fitted on the training set: with and without outliers.  Visually which model fits the test set data more closely?



```python
outliers_x = outliers_train.loc[outlier_indices]['X']
outliers_y = outliers_train.loc[outlier_indices]['Y']

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].grid(True, lw=1.75, ls='--', alpha=0.15)
ax[0].scatter(x_train, y_train, c='g', alpha=0.5, label='Training Set')
ax[0].scatter(outliers_x, outliers_y, c='r', alpha=0.5, label='Outliers')
ax[0].plot(x_train, results_sm.fittedvalues, c='g', alpha=0.5, label='Raw Regression Line')
ax[0].plot(x_train_clean, results_sm_clean.fittedvalues, c='b', alpha=0.5, label='Clean Regression Line')
ax[0].set_title(r'Training Set')
ax[0].set_xlabel(r'$X$')
ax[0].set_ylabel(r'$Y$')
ax[0].legend(loc='upper right')

ax[1].grid(True, lw=1.75, ls='--', alpha=0.15)
ax[1].scatter(x_test, y_test, c='y', alpha=0.5, label='Testing Set')
ax[1].plot(x_train, results_sm.fittedvalues, c='g', alpha=0.5, label='Raw Regression Line')
ax[1].plot(x_train_clean, results_sm_clean.fittedvalues, c='b', alpha=0.5, label='Clean Regression Line')
ax[1].set_title(r'Testing Set')
ax[1].set_xlabel(r'$X$')
ax[1].set_ylabel(r'$Y$')
ax[1].legend(loc='upper right');
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_46_0.png)


Visually the regression line without outliers fit the test set data more closely.  

**4.6**. Calculate the $R^2$ score for the `OutlierOLSModel` and the `OutlierFreeSimpleModel` on the test set data.  Which model produces a better $R^2$ score?



```python
X_test = sm.add_constant(x_test)
r2_raw = r2_score(y_test, results_sm.predict(X_test))
r2_clean = r2_score(y_test, results_sm_clean.predict(X_test))

print("OutlierOLSModel R-squared: {:.3f}".format(r2_raw))
print("OutlierFreeSimpleModel R-squared: {:.3f}".format(r2_clean))
```


    OutlierOLSModel R-squared: 0.341
    OutlierFreeSimpleModel R-squared: 0.453


Conclusion: OutlierFreeSimpleModel is better than OutlierOLSModel.

**4.7**. One potential problem with the brute force outlier detection approach in 4.3 and the heuristic algorithm you constructed 4.4 is that they assume prior knowledge of the number of outliers.  In general you can't expect to know ahead of time the number of outliers in your dataset.  Alter the algorithm you constructed in 4.4 to create a more general heuristic (i.e. one which doesn't presuppose the number of outliers) for finding outliers in your dataset.  Store your algorithm in the function `find_outliers_general`.  It should take the parameters `dataset_x` and `dataset_y` representing your features and response variable values (make sure your response variable is stored as a numpy column vector).  It can take additional parameters as long as they have default values set.  The return value should be the list `outlier_indices` representing the indices of the outliers in the original datasets you passed in (in the order of 'severity').  Remove the outliers that your algorithm identified, use `statsmodels` to create a Linear Regression model on the remaining training set data, and store your model in the variable `OutlierFreeGeneralModel`.




```python
## Logic: Drop outliers until R-squared starts to decrease.  
## Could add a minimium R-squared improvement requirment. If less than the cutoff, stop the algorithm.
def find_outliers_general(dataset_x, dataset_y, n_trails):
    dataset_x_raw = dataset_x
    dataset_y_raw = dataset_y
    
    r2_value = []
    outliers = []
    
    for _ in range(n_trails):
        X_train = sm.add_constant(dataset_x)
        OutlierOLSModel = sm.OLS(dataset_y, X_train)
        results_sm = OutlierOLSModel.fit()
        
        ## if r-squared starts to decrease, exit the function 
        if r2_value and results_sm.rsquared < r2_value[-1]:
            break
    
        r2_value.append(results_sm.rsquared)
        
        df = pd.DataFrame({'X': dataset_x, 
                           'Y': dataset_y, 
                           'resid': results_sm.resid, 
                           'resid_abs': abs(results_sm.resid)
                          })
        
        df.sort_values(by=['resid_abs'], ascending=False, inplace=True)
        
        outlier_indices = list(df.iloc[:1].index)
        outliers += outlier_indices
        df.drop(outlier_indices, inplace=True)
        dataset_x, dataset_y = df['X'], df['Y']
    
    return r2_value, outliers
```


**4.8**. Run your algorithm in 4.7 on the training set data.  
1. What outliers does it identify?
2. How do those outliers compare to the outliers you found in 4.4?
3. How does the general outlier-free Linear Regression model you created in 4.7 perform compared to the simple one in 4.4?



```python
r2_value, outliers = find_outliers_general(x_train, y_train, x_train.shape[0])

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.grid(True, lw=1.75, ls='--', alpha=0.15)
ax.plot(r2_value, c='b', alpha=0.5)
ax.set_title(r'R Squared vs Number of Outliers droped', fontsize=14)
ax.set_xlabel(r'Number of Outliers droped', fontsize=14)
ax.set_ylabel(r'R Squared', fontsize=14);
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_54_0.png)




```python
outliers_x_gen = outliers_train.loc[outliers]['X']
outliers_y_gen = outliers_train.loc[outliers]['Y']
outliers_train_gen = outliers_train.drop(outliers)

x_train_gen, y_train_gen = outliers_train_gen['X'], outliers_train_gen['Y']
X = sm.add_constant(x_train_gen)
OutlierFreeGeneralModel = sm.OLS(y_train_gen, X)
results_sm_gen = OutlierFreeGeneralModel.fit()
print(results_sm_gen.summary())
```


                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      Y   R-squared:                       0.488
    Model:                            OLS   Adj. R-squared:                  0.475
    Method:                 Least Squares   F-statistic:                     38.14
    Date:                Sun, 08 Sep 2019   Prob (F-statistic):           2.68e-07
    Time:                        09:54:45   Log-Likelihood:                -245.20
    No. Observations:                  42   AIC:                             494.4
    Df Residuals:                      40   BIC:                             497.9
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -5.9159     13.134     -0.450      0.655     -32.461      20.629
    X             83.5494     13.529      6.176      0.000      56.207     110.892
    ==============================================================================
    Omnibus:                        2.111   Durbin-Watson:                   1.520
    Prob(Omnibus):                  0.348   Jarque-Bera (JB):                1.278
    Skew:                           0.075   Prob(JB):                        0.528
    Kurtosis:                       2.159   Cond. No.                         1.05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.grid(True, lw=1.75, ls='--', alpha=0.15)
ax.scatter(x_train, y_train, c='g', alpha=0.5, label='Training Set')
ax.scatter(outliers_x_gen, outliers_y_gen, c='r', alpha=0.5, label='Outliers')
ax.plot(x_train, results_sm.fittedvalues, c='g', alpha=0.5, label='Raw Regression Line')
ax.plot(x_train_gen, results_sm_gen.fittedvalues, c='b', alpha=0.5, label='Regression Line without 10 outliers')
ax.set_title(r'Training Set - Regression $Y = \beta_0 + \beta_1 X + \epsilon$')
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.legend(loc='upper right');
```



![png](cs109a_hw2_web_files/cs109a_hw2_web_56_0.png)




```python
## Output results
X_test = sm.add_constant(x_test)
r2_clean = r2_score(y_test, results_sm_clean.predict(X_test))
r2_gen = r2_score(y_test, results_sm_gen.predict(X_test))
print("OutlierFreeSimpleModel R-squared: {:.3f}".format(r2_clean))
print("OutlierFreeGeneralModel R-squared: {:.3f}".format(r2_gen))
```


    OutlierFreeSimpleModel R-squared: 0.453
    OutlierFreeGeneralModel R-squared: 0.454


**4.8.1** What outliers does it identify?

Answer: The following indices are identified as outliers: [50, 51, 52, 1, 14, 28, 5, 35, 24, 20, 7]

**4.8.2** How do those outliers compare to the outliers you found in 4.4?

Answer: There are 10 outliers detected in the general method, because droping the 11th outliers will cause the r-squared to decrease, while 4.4 assumes that there are 3 outliers in total with the above scatter plot showing the distribution. 

**4.8.3** How does the general outlier-free Linear Regression model you created in 4.7 perform compared to the simple one in 4.4?

Answer: General model in 4.7 is slightly better than the model in 4.4, with R-squared of 0.453556 vs 0.452957.

