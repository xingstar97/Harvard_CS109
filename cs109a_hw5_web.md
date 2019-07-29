---
title: H5
notebook: cs109a_hw5_web.ipynb
nav_include: 7
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

import statsmodels.api as sm
from statsmodels.api import OLS

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import math
from scipy.special import gamma

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from IPython.display import display

```


<div class='theme'> Cancer Classification from Gene Expressions </div>

In this problem, we will build a classification model to distinguish between two related classes of cancer, acute lymphoblastic leukemia (ALL) and acute myeloid leukemia (AML), using gene expression measurements. The data set is provided in the file `data/dataset_hw5_1.csv`. Each row in this file corresponds to a tumor tissue sample from a patient with one of the two forms of Leukemia. The first column contains the cancer type, with 0 indicating the ALL class and 1 indicating the AML class. Columns 2-7130 contain expression levels of 7129 genes recorded from each tissue sample. 

In the following questions, we will use linear and logistic regression to build classification models for this data set. We will also use Principal Components Analysis (PCA) to reduce its dimensions. 



## Data Exploration

First step is to split the observations into an approximate 50-50 train-test split. Below is some code to do this for you (we want to make sure everyone has the same splits).



```python
np.random.seed(9002)
df = pd.read_csv('data/dataset_hw5_1.csv')
msk = np.random.rand(len(df)) < 0.5
data_train = df[msk]
data_test = df[~msk]
```


**1.1:** Take a peek at your training set: you should notice the severe differences in the measurements from one gene to the next (some are negative, some hover around zero, and some are well into the thousands).  To account for these differences in scale and variability, normalize each predictor to vary between 0 and 1.




```python
display(data_train.head())
display(data_test.head())
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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-214</td>
      <td>-153</td>
      <td>-58</td>
      <td>88</td>
      <td>-295</td>
      <td>-558</td>
      <td>199</td>
      <td>-176</td>
      <td>252</td>
      <td>...</td>
      <td>185</td>
      <td>511</td>
      <td>-125</td>
      <td>389</td>
      <td>-37</td>
      <td>793</td>
      <td>329</td>
      <td>36</td>
      <td>191</td>
      <td>-37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-106</td>
      <td>-125</td>
      <td>-76</td>
      <td>168</td>
      <td>-230</td>
      <td>-284</td>
      <td>4</td>
      <td>-122</td>
      <td>70</td>
      <td>...</td>
      <td>156</td>
      <td>649</td>
      <td>57</td>
      <td>504</td>
      <td>-26</td>
      <td>250</td>
      <td>314</td>
      <td>14</td>
      <td>56</td>
      <td>-25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>-67</td>
      <td>-93</td>
      <td>84</td>
      <td>25</td>
      <td>-179</td>
      <td>-323</td>
      <td>-135</td>
      <td>-127</td>
      <td>-2</td>
      <td>...</td>
      <td>48</td>
      <td>224</td>
      <td>60</td>
      <td>194</td>
      <td>-10</td>
      <td>291</td>
      <td>41</td>
      <td>8</td>
      <td>-2</td>
      <td>-80</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>-476</td>
      <td>-213</td>
      <td>-18</td>
      <td>301</td>
      <td>-403</td>
      <td>-394</td>
      <td>-42</td>
      <td>-144</td>
      <td>98</td>
      <td>...</td>
      <td>241</td>
      <td>1214</td>
      <td>127</td>
      <td>255</td>
      <td>50</td>
      <td>1701</td>
      <td>1108</td>
      <td>61</td>
      <td>525</td>
      <td>-83</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>-81</td>
      <td>-150</td>
      <td>-119</td>
      <td>78</td>
      <td>-152</td>
      <td>-340</td>
      <td>-36</td>
      <td>-141</td>
      <td>96</td>
      <td>...</td>
      <td>186</td>
      <td>573</td>
      <td>-57</td>
      <td>694</td>
      <td>-19</td>
      <td>636</td>
      <td>205</td>
      <td>17</td>
      <td>127</td>
      <td>-13</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7130 columns</p>
</div>



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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>-135</td>
      <td>-114</td>
      <td>265</td>
      <td>12</td>
      <td>-419</td>
      <td>-585</td>
      <td>158</td>
      <td>-253</td>
      <td>49</td>
      <td>...</td>
      <td>240</td>
      <td>835</td>
      <td>218</td>
      <td>174</td>
      <td>-110</td>
      <td>627</td>
      <td>170</td>
      <td>-50</td>
      <td>126</td>
      <td>-91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>-72</td>
      <td>-144</td>
      <td>238</td>
      <td>55</td>
      <td>-399</td>
      <td>-551</td>
      <td>131</td>
      <td>-179</td>
      <td>126</td>
      <td>...</td>
      <td>30</td>
      <td>819</td>
      <td>-178</td>
      <td>151</td>
      <td>-18</td>
      <td>1140</td>
      <td>482</td>
      <td>10</td>
      <td>369</td>
      <td>-42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-413</td>
      <td>-260</td>
      <td>7</td>
      <td>-2</td>
      <td>-541</td>
      <td>-790</td>
      <td>-275</td>
      <td>-463</td>
      <td>70</td>
      <td>...</td>
      <td>289</td>
      <td>629</td>
      <td>-86</td>
      <td>302</td>
      <td>23</td>
      <td>1798</td>
      <td>446</td>
      <td>59</td>
      <td>781</td>
      <td>20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>-92</td>
      <td>-119</td>
      <td>-31</td>
      <td>173</td>
      <td>-233</td>
      <td>-227</td>
      <td>-49</td>
      <td>-62</td>
      <td>13</td>
      <td>...</td>
      <td>213</td>
      <td>583</td>
      <td>3</td>
      <td>530</td>
      <td>-39</td>
      <td>696</td>
      <td>302</td>
      <td>24</td>
      <td>74</td>
      <td>-11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>-107</td>
      <td>-72</td>
      <td>-126</td>
      <td>149</td>
      <td>-205</td>
      <td>-284</td>
      <td>-166</td>
      <td>-185</td>
      <td>1</td>
      <td>...</td>
      <td>120</td>
      <td>722</td>
      <td>20</td>
      <td>332</td>
      <td>-5</td>
      <td>195</td>
      <td>59</td>
      <td>31</td>
      <td>116</td>
      <td>-18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7130 columns</p>
</div>




```python
min_vals = data_train.min()
max_vals = data_train.max()

data_train = (data_train - min_vals)/(max_vals - min_vals)
data_test = (data_test - min_vals)/(max_vals - min_vals)
```




```python
display(data_train.head())
display(data_test.head())
display(data_train.describe())
display(data_test.describe())
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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.466192</td>
      <td>0.739726</td>
      <td>0.255814</td>
      <td>0.246154</td>
      <td>0.433190</td>
      <td>0.240418</td>
      <td>0.880427</td>
      <td>0.625850</td>
      <td>0.928074</td>
      <td>...</td>
      <td>0.385445</td>
      <td>0.268542</td>
      <td>0.398126</td>
      <td>0.161897</td>
      <td>0.677778</td>
      <td>0.323241</td>
      <td>0.322609</td>
      <td>0.751381</td>
      <td>0.069457</td>
      <td>0.381720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.658363</td>
      <td>0.794521</td>
      <td>0.213953</td>
      <td>0.421978</td>
      <td>0.573276</td>
      <td>0.717770</td>
      <td>0.741637</td>
      <td>0.748299</td>
      <td>0.505800</td>
      <td>...</td>
      <td>0.307278</td>
      <td>0.356777</td>
      <td>0.824356</td>
      <td>0.206978</td>
      <td>0.718519</td>
      <td>0.081478</td>
      <td>0.309565</td>
      <td>0.629834</td>
      <td>0.027597</td>
      <td>0.446237</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.727758</td>
      <td>0.857143</td>
      <td>0.586047</td>
      <td>0.107692</td>
      <td>0.683190</td>
      <td>0.649826</td>
      <td>0.642705</td>
      <td>0.736961</td>
      <td>0.338747</td>
      <td>...</td>
      <td>0.016173</td>
      <td>0.085038</td>
      <td>0.831382</td>
      <td>0.085457</td>
      <td>0.777778</td>
      <td>0.099733</td>
      <td>0.072174</td>
      <td>0.596685</td>
      <td>0.009612</td>
      <td>0.150538</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.622309</td>
      <td>0.348837</td>
      <td>0.714286</td>
      <td>0.200431</td>
      <td>0.526132</td>
      <td>0.708897</td>
      <td>0.698413</td>
      <td>0.570766</td>
      <td>...</td>
      <td>0.536388</td>
      <td>0.718031</td>
      <td>0.988290</td>
      <td>0.109369</td>
      <td>1.000000</td>
      <td>0.727516</td>
      <td>1.000000</td>
      <td>0.889503</td>
      <td>0.173023</td>
      <td>0.134409</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.702847</td>
      <td>0.745597</td>
      <td>0.113953</td>
      <td>0.224176</td>
      <td>0.741379</td>
      <td>0.620209</td>
      <td>0.713167</td>
      <td>0.705215</td>
      <td>0.566125</td>
      <td>...</td>
      <td>0.388140</td>
      <td>0.308184</td>
      <td>0.557377</td>
      <td>0.281458</td>
      <td>0.744444</td>
      <td>0.253339</td>
      <td>0.214783</td>
      <td>0.646409</td>
      <td>0.049612</td>
      <td>0.510753</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7130 columns</p>
</div>



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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.606762</td>
      <td>0.816047</td>
      <td>1.006977</td>
      <td>0.079121</td>
      <td>0.165948</td>
      <td>0.193380</td>
      <td>0.851246</td>
      <td>0.451247</td>
      <td>0.457077</td>
      <td>...</td>
      <td>0.533693</td>
      <td>0.475703</td>
      <td>1.201405</td>
      <td>0.077617</td>
      <td>0.407407</td>
      <td>0.249332</td>
      <td>0.184348</td>
      <td>0.276243</td>
      <td>0.049302</td>
      <td>0.091398</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.718861</td>
      <td>0.757339</td>
      <td>0.944186</td>
      <td>0.173626</td>
      <td>0.209052</td>
      <td>0.252613</td>
      <td>0.832028</td>
      <td>0.619048</td>
      <td>0.635731</td>
      <td>...</td>
      <td>-0.032345</td>
      <td>0.465473</td>
      <td>0.274005</td>
      <td>0.068601</td>
      <td>0.748148</td>
      <td>0.477738</td>
      <td>0.455652</td>
      <td>0.607735</td>
      <td>0.124651</td>
      <td>0.354839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.112100</td>
      <td>0.530333</td>
      <td>0.406977</td>
      <td>0.048352</td>
      <td>-0.096983</td>
      <td>-0.163763</td>
      <td>0.543060</td>
      <td>-0.024943</td>
      <td>0.505800</td>
      <td>...</td>
      <td>0.665768</td>
      <td>0.343990</td>
      <td>0.489461</td>
      <td>0.127793</td>
      <td>0.900000</td>
      <td>0.770703</td>
      <td>0.424348</td>
      <td>0.878453</td>
      <td>0.252403</td>
      <td>0.688172</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.683274</td>
      <td>0.806262</td>
      <td>0.318605</td>
      <td>0.432967</td>
      <td>0.566810</td>
      <td>0.817073</td>
      <td>0.703915</td>
      <td>0.884354</td>
      <td>0.373550</td>
      <td>...</td>
      <td>0.460916</td>
      <td>0.314578</td>
      <td>0.697892</td>
      <td>0.217170</td>
      <td>0.670370</td>
      <td>0.280053</td>
      <td>0.299130</td>
      <td>0.685083</td>
      <td>0.033178</td>
      <td>0.521505</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.656584</td>
      <td>0.898239</td>
      <td>0.097674</td>
      <td>0.380220</td>
      <td>0.627155</td>
      <td>0.717770</td>
      <td>0.620641</td>
      <td>0.605442</td>
      <td>0.345708</td>
      <td>...</td>
      <td>0.210243</td>
      <td>0.403453</td>
      <td>0.737705</td>
      <td>0.139553</td>
      <td>0.796296</td>
      <td>0.056990</td>
      <td>0.087826</td>
      <td>0.723757</td>
      <td>0.046202</td>
      <td>0.483871</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7130 columns</p>
</div>



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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>40.00000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>...</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.37500</td>
      <td>0.640347</td>
      <td>0.719472</td>
      <td>0.369477</td>
      <td>0.512253</td>
      <td>0.529472</td>
      <td>0.550653</td>
      <td>0.654253</td>
      <td>0.581803</td>
      <td>0.535615</td>
      <td>...</td>
      <td>0.385916</td>
      <td>0.351822</td>
      <td>0.656206</td>
      <td>0.173726</td>
      <td>0.698519</td>
      <td>0.334762</td>
      <td>0.283348</td>
      <td>0.615608</td>
      <td>0.162620</td>
      <td>0.404570</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.49029</td>
      <td>0.182889</td>
      <td>0.186767</td>
      <td>0.237206</td>
      <td>0.243956</td>
      <td>0.231075</td>
      <td>0.214332</td>
      <td>0.216589</td>
      <td>0.228836</td>
      <td>0.231285</td>
      <td>...</td>
      <td>0.234882</td>
      <td>0.195379</td>
      <td>0.217202</td>
      <td>0.154501</td>
      <td>0.201592</td>
      <td>0.204001</td>
      <td>0.171412</td>
      <td>0.200119</td>
      <td>0.202527</td>
      <td>0.199781</td>
    </tr>
    <tr>
      <th>min</th>
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
      <td>...</td>
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
      <th>25%</th>
      <td>0.00000</td>
      <td>0.596530</td>
      <td>0.631115</td>
      <td>0.201744</td>
      <td>0.325824</td>
      <td>0.386853</td>
      <td>0.408101</td>
      <td>0.547153</td>
      <td>0.484127</td>
      <td>0.342807</td>
      <td>...</td>
      <td>0.247305</td>
      <td>0.213235</td>
      <td>0.556792</td>
      <td>0.104175</td>
      <td>0.602778</td>
      <td>0.210597</td>
      <td>0.200870</td>
      <td>0.506906</td>
      <td>0.057054</td>
      <td>0.243280</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.00000</td>
      <td>0.653025</td>
      <td>0.745597</td>
      <td>0.323256</td>
      <td>0.553846</td>
      <td>0.584052</td>
      <td>0.542683</td>
      <td>0.683986</td>
      <td>0.634921</td>
      <td>0.573086</td>
      <td>...</td>
      <td>0.370620</td>
      <td>0.337596</td>
      <td>0.715457</td>
      <td>0.143865</td>
      <td>0.737037</td>
      <td>0.302760</td>
      <td>0.281739</td>
      <td>0.607735</td>
      <td>0.085271</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.00000</td>
      <td>0.731762</td>
      <td>0.844423</td>
      <td>0.500000</td>
      <td>0.720330</td>
      <td>0.683728</td>
      <td>0.713850</td>
      <td>0.753381</td>
      <td>0.739796</td>
      <td>0.725058</td>
      <td>...</td>
      <td>0.498652</td>
      <td>0.407289</td>
      <td>0.813232</td>
      <td>0.195904</td>
      <td>0.856481</td>
      <td>0.434328</td>
      <td>0.342391</td>
      <td>0.708564</td>
      <td>0.176434</td>
      <td>0.512097</td>
    </tr>
    <tr>
      <th>max</th>
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
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 7130 columns</p>
</div>



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
      <th>Cancer_type</th>
      <th>AFFX-BioB-5_at</th>
      <th>AFFX-BioB-M_at</th>
      <th>AFFX-BioB-3_at</th>
      <th>AFFX-BioC-5_at</th>
      <th>AFFX-BioC-3_at</th>
      <th>AFFX-BioDn-5_at</th>
      <th>AFFX-BioDn-3_at</th>
      <th>AFFX-CreX-5_at</th>
      <th>AFFX-CreX-3_at</th>
      <th>...</th>
      <th>U48730_at</th>
      <th>U58516_at</th>
      <th>U73738_at</th>
      <th>X06956_at</th>
      <th>X16699_at</th>
      <th>X83863_at</th>
      <th>Z17240_at</th>
      <th>L49218_f_at</th>
      <th>M71243_f_at</th>
      <th>Z78285_f_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>...</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.303030</td>
      <td>0.641055</td>
      <td>0.733737</td>
      <td>0.371388</td>
      <td>0.409524</td>
      <td>0.512409</td>
      <td>0.479516</td>
      <td>0.744937</td>
      <td>0.606473</td>
      <td>0.533502</td>
      <td>...</td>
      <td>0.328188</td>
      <td>0.387352</td>
      <td>0.664112</td>
      <td>0.141442</td>
      <td>0.679686</td>
      <td>0.297215</td>
      <td>0.322661</td>
      <td>0.597355</td>
      <td>0.120583</td>
      <td>0.429456</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.466694</td>
      <td>0.165245</td>
      <td>0.189658</td>
      <td>0.335398</td>
      <td>0.239065</td>
      <td>0.298177</td>
      <td>0.309424</td>
      <td>0.174231</td>
      <td>0.222876</td>
      <td>0.187403</td>
      <td>...</td>
      <td>0.271523</td>
      <td>0.194909</td>
      <td>0.193956</td>
      <td>0.138439</td>
      <td>0.180964</td>
      <td>0.194876</td>
      <td>0.186667</td>
      <td>0.296720</td>
      <td>0.179463</td>
      <td>0.233766</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.112100</td>
      <td>0.174168</td>
      <td>-0.562791</td>
      <td>-0.026374</td>
      <td>-0.096983</td>
      <td>-0.198606</td>
      <td>0.489680</td>
      <td>-0.024943</td>
      <td>0.153132</td>
      <td>...</td>
      <td>-0.269542</td>
      <td>0.021739</td>
      <td>0.257611</td>
      <td>0.014504</td>
      <td>0.266667</td>
      <td>0.029386</td>
      <td>0.035652</td>
      <td>-0.591160</td>
      <td>-0.010853</td>
      <td>-0.440860</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.560498</td>
      <td>0.618395</td>
      <td>0.255814</td>
      <td>0.232967</td>
      <td>0.312500</td>
      <td>0.240418</td>
      <td>0.620641</td>
      <td>0.519274</td>
      <td>0.406032</td>
      <td>...</td>
      <td>0.164420</td>
      <td>0.268542</td>
      <td>0.578454</td>
      <td>0.076833</td>
      <td>0.551852</td>
      <td>0.203028</td>
      <td>0.184348</td>
      <td>0.508287</td>
      <td>0.046202</td>
      <td>0.349462</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.683274</td>
      <td>0.767123</td>
      <td>0.374419</td>
      <td>0.373626</td>
      <td>0.566810</td>
      <td>0.496516</td>
      <td>0.716726</td>
      <td>0.625850</td>
      <td>0.494200</td>
      <td>...</td>
      <td>0.239892</td>
      <td>0.350384</td>
      <td>0.662763</td>
      <td>0.103489</td>
      <td>0.677778</td>
      <td>0.269368</td>
      <td>0.322609</td>
      <td>0.607735</td>
      <td>0.074729</td>
      <td>0.451613</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.749110</td>
      <td>0.872798</td>
      <td>0.504651</td>
      <td>0.558242</td>
      <td>0.698276</td>
      <td>0.736934</td>
      <td>0.839146</td>
      <td>0.752834</td>
      <td>0.635731</td>
      <td>...</td>
      <td>0.463612</td>
      <td>0.475703</td>
      <td>0.768150</td>
      <td>0.163073</td>
      <td>0.796296</td>
      <td>0.330365</td>
      <td>0.423478</td>
      <td>0.751381</td>
      <td>0.090853</td>
      <td>0.575269</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.873665</td>
      <td>1.013699</td>
      <td>1.116279</td>
      <td>0.923077</td>
      <td>1.314655</td>
      <td>0.954704</td>
      <td>1.243416</td>
      <td>0.970522</td>
      <td>0.928074</td>
      <td>...</td>
      <td>0.846361</td>
      <td>0.872123</td>
      <td>1.201405</td>
      <td>0.769502</td>
      <td>1.007407</td>
      <td>0.838379</td>
      <td>0.824348</td>
      <td>1.187845</td>
      <td>0.791628</td>
      <td>0.795699</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 7130 columns</p>
</div>


From the above statistics of train dataframe, we can see that training dataset is normalised. Test dataset still has some negative values, since we are useing max and min from train dataset to normalise the test set. <br/>

**1.2:** Notice that the results training set contains significantly more predictors than observations. Do you foresee a problem in fitting a classification model to such a data set?



```python
data_train.shape
```





    (40, 7130)



The training set is improper as it contains many more columns compared to number of samples. If we fit models to the given dataset, they will be highly overfitted. This is called the curse of dimensionality.

**1.3:** Lets explore a few of the genes and see how well they discriminate between cancer classes. Create a single figure with four subplots arranged in a 2x2 grid. Consider the following four genes: `D29963_at`, `M23161_at`, `hum_alu_at`, and `AFFX-PheX-5_at`. For each gene overlay two histograms of the gene expression values on one of the subplots, one histogram for each cancer type. Does it appear that any of these genes discriminate between the two classes well? How are you able to tell?



```python
alpha=0.5

fig, ax = plt.subplots(2, 2, figsize=(16, 8))

cancer_type_0 = data_train[data_train['Cancer_type'] == 0]
cancer_type_1 = data_train[data_train['Cancer_type'] == 1]

ax[0, 0].hist(cancer_type_0['D29963_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[0, 0].hist(cancer_type_1['D29963_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[0, 0].set_title('D29963_at')
ax[0, 0].set_xlabel('Normalized Gene Measure')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].legend()

ax[0, 1].hist(cancer_type_0['M23161_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[0, 1].hist(cancer_type_1['M23161_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[0, 1].set_title('M23161_at')
ax[0, 1].set_xlabel('Normalized Gene Measure')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].legend()

ax[1, 0].hist(cancer_type_0['hum_alu_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[1, 0].hist(cancer_type_1['hum_alu_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[1, 0].set_title('hum_alu_at')
ax[1, 0].set_xlabel('Normalized Gene Measure')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].legend()

ax[1, 1].hist(cancer_type_0['AFFX-PheX-5_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[1, 1].hist(cancer_type_1['AFFX-PheX-5_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[1, 1].set_title('AFFX-PheX-5_at')
ax[1, 1].set_xlabel('Normalized Gene Measure')
ax[1, 1].set_ylabel('Frequency')
ax[1, 1].legend();
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_15_0.png)


From the above histograms, we may argue that `D29963_at` and `hum_alu_at` have slightly disjointed regions in the graph and may be singly able to differentiate between the cancer types. For the other two genes, the histograms for both cancer types seem to overlap and don't have distinctive regions. <br/> <br/>

If in any of the above plots, the frequency level of each cancer type was different or the gene measure was in different ranges, we could argue that that a gene is able to distinguish between different cancer types well. This may be said for different frequency levels of `D29963_at` but doesn't seem true in any other case.

It is also useful to note the shape of the individual distributions as well. For instance, `hum_alu_at` has a very bimodal distribution for type 1, whereas for 0 it is more unimodal. The same can be said for `M23161_at` to a lesser extent. Loosely speaking, the characteristics of the distributions could be used to compare the odds of a given gene expression value indicating 0 vs 1. 

Finally, it is worth noting in our solution that `density`/`normed` is not ensuring that the result matches the **true** continuous PDF, but rather simply scaling the counts of the values to turn it into a valid discrete density function determined solely by our samples. This helps us compare the two cancer types, as the number of samples from each cancer type is different. 

**1.4:** Since our data has dimensions that are not easily visualizable, we want to reduce the dimensionality of the data to make it easier to visualize. Using PCA, find the top two principal components for the gene expression data. Generate a scatter plot using these principal components, highlighting the two cancer types in different colors. How well do the top two principal components discriminate between the two classes? How much of the variance within the data do these two principal components explain?



```python
x_train, y_train = data_train.iloc[:,1:], data_train.iloc[:,0]
x_test, y_test = data_test.iloc[:,1:], data_test.iloc[:,0]

print("x train shape:", x_train.shape, "test shape:", x_test.shape)

pca = PCA(n_components = 2).fit(x_train)
pca_x = pca.transform(x_train)

print("pca shape:", pca_x.shape)
```


    x train shape: (40, 7129) test shape: (33, 7129)
    pca shape: (40, 2)




```python
pca_df = pd.DataFrame(pca_x, columns=['PCA1', 'PCA2'])
pca_df['Cancer_type'] = y_train.values

sns.lmplot(x="PCA1", y="PCA2", hue='Cancer_type', data=pca_df, fit_reg=False)
plt.title('Scatter plot of top 2 components of different types of cancer data');
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_19_0.png)




```python
print( "Variance explained:", np.sum(pca.explained_variance_ratio_))
```


    Variance explained: 0.2731782945208866


The first two principal components only show a limited capacity to discriminate between the two cancer types. As we can see in the plot, the two types often occupy the same regions in this "PC space." This is probably due to fact that only 27% of the variance can be explained by the first two PCs, so there is potentially more information we could exploit to classify the cancer type. 

**1.5** Plot the cumulative variance explained in the feature set as a function of the number of PCA-components (up to the first 50 components).  Do you feel 2 components is enough, and if not, how many components would you choose to consider?  Justify your choice in 3 or fewer sentences.  Finally, determine how many components are needed to explain at least 90% of the variability in the feature set.



```python
var_explained = []
total_comp = 40
pca = PCA(n_components = total_comp).fit(x_train)

plt.plot(np.linspace(1, total_comp, total_comp), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('variance explained')
plt.title('Cumulative variance explained',fontsize=15)

print("number of components that explain at least 90% of the variance=",\
    len(np.where(np.cumsum(pca.explained_variance_ratio_)<=0.9)[0])+1);
```


    number of components that explain at least 90% of the variance= 29



![png](cs109a_hw5_web_files/cs109a_hw5_web_23_1.png)


The number of principal components cannot be greater than the number of samples or the number of features. So in this case we can only get at most 40 PCs. As shown above, 29 components are needed to explain at least 90% of the variability in the feature set.

Two components are probably not enough in this context, since they can only explain 27% of the variance in the data. However, the number of components to consider ultimately depends on how we plan to use the transformed data. If we simply want to visualize the classes, then it is clear we cannot use more than three components. If we want to train a classifier on the transformed data (as we do later on), then we could consider a series of different numbers of PCs, and observe how the classifiation accuracy changes based on the number used. We could also consider the point of diminishing returns of adding new PCs, as demonstrated below.



```python
plt.scatter(range(1, 41), pca.explained_variance_ratio_)
plt.xlabel('index of components')
plt.ylabel('variance explained')
plt.title('Variance explained by each component',fontsize=15)
```





    Text(0.5, 1.0, 'Variance explained by each component')




![png](cs109a_hw5_web_files/cs109a_hw5_web_25_1.png)


From the plot above we see approximately that the first 11 components explain relatively high variance, and the components afterward each explain similar amounts of variance. If we choose the first 11 components based on this observation, we would explain more than 60% of total variance.



```python
print("The variance explained by first 11 components is",\
    np.sum(pca.explained_variance_ratio_[:11]))
```


    The variance explained by first 11 components is 0.604403665357437


## Linear Regression vs. Logistic Regression 

In class we discussed how to use both linear regression and logistic regression for classification. For this question, you will work with a single gene predictor, `D29963_at`, to explore these two methods.


**2.1:** Fit a simple linear regression model to the training set using the single gene predictor D29963_at to predict cancer type and plot the histogram of predicted values. We could interpret the scores predicted by the regression model for a patient as an estimate of the probability that the patient has Cancer_type=1 (AML). Is there a problem with this interpretation?



```python
single_x_train = sm.add_constant(x_train['D29963_at'])
single_x_test = sm.add_constant(x_test['D29963_at'])

print("shapes of x_train and x_test", single_x_train.shape, single_x_test.shape)

regr = OLS(y_train, single_x_train).fit()
y_train_pred = regr.predict(single_x_train)
y_test_pred = regr.predict(single_x_test)

fig = plt.figure();
host = fig.add_subplot(111)
par1 = host.twinx()

host.set_ylabel("Probability")
par1.set_ylabel("Class")

host.plot(x_train['D29963_at'], y_train_pred, '-');
host.plot(x_train['D29963_at'], y_train, 's');
host.set_xlabel('Normalized D29963_at')
host.set_ylabel('Probability of being ALM')

labels = ['ALL', 'ALM'];

par1.set_yticks( [0.082, 0.81]);
par1.set_yticklabels(labels);
```


    shapes of x_train and x_test (40, 2) (33, 2)



![png](cs109a_hw5_web_files/cs109a_hw5_web_30_1.png)




```python
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
axs.hist(y_train_pred, alpha=0.5)
axs.set_xlabel("Response Values of Trained Model")
axs.set_ylabel("Count")
axs.set_title("Output Probability Histogram on Training Data")
plt.show()
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_31_0.png)


If we take the regression prediction as probability of having Cancer Type 1, it is not clear how we will determine the presence of having cancer type 1 - is it 1-probability of Cancer type 1? We will also not be able to interpret response values $> 1$ or $< 0$.

**2.2:** The fitted linear regression model can be converted to a classification model (i.e. a model that predicts one of two binary classes 0 or 1) by classifying patients with predicted score greater than 0.5 into `Cancer_type`=1, and the others into the `Cancer_type`=0. Evaluate the classification accuracy of the obtained classification model on both the training and test sets.



```python
train_score = accuracy_score(y_train, y_train_pred>0.5)
test_score = accuracy_score(y_test, y_test_pred>0.5)
print("train score:", train_score, ", test score:", test_score)
```


    train score: 0.8 , test score: 0.7575757575757576


We get 80% classification accuracy for training set and 75.75% for test set.

**2.3:**  Next, fit a simple logistic regression model to the training set. How do the training and test classification accuracies of this model compare with the linear regression model? If there are no substantial differences, why do you think this happens?



```python
logreg = LogisticRegression(C=100000, fit_intercept=False)
logreg.fit(single_x_train, y_train) 

y_train_pred_logreg = logreg.predict(single_x_train)
y_test_pred_logreg = logreg.predict(single_x_test)

y_train_pred_logreg_prob = logreg.predict_proba(single_x_train)[:,1]
y_test_pred_logreg_prob = logreg.predict_proba(single_x_test)[:,1]

train_score_logreg = accuracy_score(y_train, y_train_pred_logreg)
test_score_logreg = accuracy_score(y_test, y_test_pred_logreg)

print("train score:", train_score_logreg, "test score:", test_score_logreg)
```


    train score: 0.8 test score: 0.7575757575757576


Logistic regression gave the same accuracy as Linear Regression model. Classification are going to always be similar (especially in simple models) with few points. 

**2.4:** Create a figure with 4 items displayed on the same plot:
- the quantitative response from the linear regression model as a function of the gene predictor `D29963_at`.
- the predicted probabilities of the logistic regression model as a function of the gene predictor `D29963_at`.  
- the true binary response for the test set points for both models in the same plot. 
- a horizontal line at $y=0.5$. 

Based on these plots, does one of the models appear better suited for binary classification than the other?  Explain in 3 sentences or fewer. 




```python
plt.figure(figsize=(12,8))
sort_index = np.argsort(x_test['D29963_at'].values)

plt.scatter(x_test['D29963_at'].iloc[sort_index], y_test.iloc[sort_index], color='black', label = 'True Response')

plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred.iloc[sort_index], color='red', alpha=0.3, \
         label = 'Linear Regression Predictions')


plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred_logreg_prob[sort_index], alpha=0.3,  \
         color='green', label = 'Logistic Regression Predictions Prob')
#plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred_logreg[sort_index], color='green', ls='-.' ,label = 'Logistic Regression Predictions')

plt.axhline(0.5, c='c')
plt.legend()
plt.title('True response v/s obtained responses')
plt.xlabel('Gene predictor value')
plt.ylabel('Cancer type response');
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_40_0.png)


Both models give similar outputs as can be observed from the plot above. However, we can see that Linear regression output goes below 0 and above 1 in some cases, whereas Logistic regression outputs are probability values between 0 and 1.

## Multiple Logistic Regression 



```python
#--------  visualize_prob

def visualize_prob(model, x, y, ax):
    # Use the model to predict probabilities for x
    y_pred = model.predict_proba(x)
    
    # Separate the predictions on the label 1 and label 0 points
    ypos = y_pred[y==1]
    yneg = y_pred[y==0]
    
    # Count the number of label 1 and label 0 points
    npos = ypos.shape[0]
    nneg = yneg.shape[0]
    
    # Plot the probabilities on a vertical line at x = 0, 
    # with the positive points in blue and negative points in red
    pos_handle = ax.plot(np.zeros((npos,1)), ypos[:,1], 'bo', label = 'Cancer Type 1')
    neg_handle = ax.plot(np.zeros((nneg,1)), yneg[:,1], 'ro', label = 'Cancer Type 0')

    # Line to mark prob 0.5
    ax.axhline(y = 0.5, color = 'k', linestyle = '--')
    
    # Add y-label and legend, do not display x-axis, set y-axis limit
    ax.set_ylabel('Probability of AML class')
    ax.legend(loc = 'best')
    ax.get_xaxis().set_visible(False)
    ax.set_ylim([0,1])
```


**3.1:** Next, fit a multiple logistic regression model with all the gene predictors from the data set. How does the classification accuracy of this model compare with the models fitted in question 2 with a single gene (on both the training and test sets)?



```python
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

all_x_train = x_train #sm.add_constant(x_train)
all_x_test = x_test # sm.add_constant(x_test)
print(all_x_train.shape, all_x_test.shape)

multi_regr = LogisticRegression(C=100000, solver = "lbfgs")
multi_regr.fit(all_x_train, y_train)

y_train_pred_multi = multi_regr.predict(all_x_train) 
y_test_pred_multi = multi_regr.predict(all_x_test)

train_score_multi = accuracy_score(y_train, y_train_pred_multi)
test_score_multi = accuracy_score(y_test, y_test_pred_multi)

print('Training set accuracy for multiple logistic regression = ', train_score_multi)
print('Test set accuracy for multiple logistic regression = ', test_score_multi)
```


    (40, 7129) (40,)
    (33, 7129) (33,)
    (40, 7129) (33, 7129)
    Training set accuracy for multiple logistic regression =  1.0
    Test set accuracy for multiple logistic regression =  1.0


<font color='red'>
The multiple logistic regression accuracy for both train and test sets are perfect at 100%. This is an improvement from the Logistic regression accuracy with single predictors. However, the moral of the story here is that (1) because of the shortage of training samples, we are overfitting to noise in the training, and (2) because of the shortage of test samples and the high dimensionality of the data, we have no way of ascertaining the generalization capacity of any trained model - which could mislead us into a false sense of security on the accuracy of the model as a whole. 
</font>


**3.2** How many of the coefficients estimated by this multiple logistic regression in the previous part are significantly different from zero at a *significance level of 5%*? Use the same value of C=100000 as before.

**Hint:** To answer this question, use *bootstrapping* with 1000 boostrap samples/iterations.  



```python
n = 1000 # Number of iterations
boot_coefs = np.zeros((all_x_train.shape[1],n)) # Create empty storage array for later use

for i in range(n):
    # Sampling WITH replacement the indices of a resampled dataset 
    sample_index = np.random.choice(range(y_train.shape[0]), size=y_train.shape[0], replace=True)

    # finding subset
    x_train_samples = all_x_train.values[sample_index]
    y_train_samples = y_train.values[sample_index]
    
    # finding logreg coefficient
    logistic_mod_boot = LogisticRegression(C=100000, fit_intercept=True, solver = "lbfgs") 
    logistic_mod_boot.fit(x_train_samples, y_train_samples) 
    boot_coefs[:,i] = logistic_mod_boot.coef_
```




```python
ci_upper = np.percentile(boot_coefs, 97.5, axis=1)
ci_lower = np.percentile(boot_coefs, 2.5, axis=1)

sig_b_ct = 0
sig_preds = []
cols = list(x_train.columns)

for i in range(len(ci_upper)):
    if ci_upper[i]<=0 or ci_lower[i]>=0:
            sig_b_ct += 1
            sig_preds.append(cols[i])

print("Significant coefficents at 5pct level = %i / %i" % (sig_b_ct, len(ci_upper)))
```


    Significant coefficents at 5pct level = 1865 / 7129


Thus, we can see that only 1865 out of 7130 predictors are significantly different from 0 at a significance level of 5%.

**3.3** Use the `visualize_prob` function provided below (or any other visualization) to visualize the probabilties predicted by the fitted multiple logistic regression model on both the training and test data sets. The function creates a visualization that places the data points on a vertical line based on the predicted probabilities, with the different cancer classes shown in different colors, and with the 0.5 threshold highlighted using a dotted horizontal line. Is there a difference in the spread of probabilities in the training and test plots? Are there data points for which the predicted probability is close to 0.5? If so, what can you say about these points?



```python
""" Plot classification model """
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

visualize_prob(multi_regr, all_x_train, y_train, ax1)
ax1.set_title('training set')

visualize_prob(multi_regr, all_x_test, y_test, ax2)
ax2.set_title('test set')
```





    Text(0.5, 1.0, 'test set')




![png](cs109a_hw5_web_files/cs109a_hw5_web_52_1.png)


<font color='red'>
Difference in spread of probability - In the training set the predicted AML values are 1 and predicted ALL values are 0. In the test set, the probabilities are spread above 0.5 but closer to 0 for points below 0.5. <br/>

There are data points for which probability is close to 0.5, indicating that it cannot be determined clearly which type of cancer the patient has. They may give false positives or false negatives.
</font>

**3.4** Open question: Comment on the classification accuracy of tain and test set? Given the results above how would you assest the generalization capacity of your trained model? What other tests would you suggest to better guard against false sense of security on the accuracy of the model as a whole. 

The answer is always cross validation 

## PCR: Principal Components Regression

High dimensional problems can lead to problematic behavior in model estimation (and make prediction on a test set worse), thus we often want to try to reduce the dimensionality of our problems. A reasonable approach to reduce the dimensionality of the data is to use PCA and fit a logistic regression model on the smallest set of principal components that explain at least 90% of the variance in the predictors.

**4.1** Fit two separate Logistic Regression models using principal components as the predictors: (1) with the number of components you selected from problem 1.5 and (2) with the number of components that explain at least 90% of the variability in the feature set. How do the classification accuracy values on both the training and tests sets compare with the models fit in question 3?



```python
pca = PCA(n_components=29) # from our result in 1.5
pca.fit(x_train)

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
```




```python
logreg_pca = LogisticRegression(C=100000, fit_intercept=False)
logreg_pca.fit(x_train_pca, y_train)

y_train_pred_logreg_pca = logreg_pca.predict(x_train_pca)
y_test_pred_logreg_pca = logreg_pca.predict(x_test_pca)

train_score_logreg_pca = accuracy_score(y_train, y_train_pred_logreg_pca)
test_score_logreg_pca = accuracy_score(y_test, y_test_pred_logreg_pca)

print('Training set accuracy for Logistic Regression with PCA = ', train_score_logreg_pca)
print('Test set accuracy for Logistic Regression with PCA = ', test_score_logreg_pca)
```


    Training set accuracy for Logistic Regression with PCA =  1.0
    Test set accuracy for Logistic Regression with PCA =  0.9393939393939394


<font color='red'>
1. The first 20 principal components explain 90.26% of the total variance in data. <br/>
2. Logistic Regression on PCA components gives 100% train accuracy and 96.96% test accuracy. The train accuracy here is same as LogReg in 3.1 but test accuracy is slightly less tha Logreg. Since PCA components only explain 90% of the data, this makes intuitive sense. <br/>
</font>

**4.2:** Use the code provided in question 3 to visualize the probabilities predicted by the fitted models on both the training and test sets. How does the spread of probabilities in these plots compare to those for the model in question 3.2? If the lower dimensional representation yields comparable predictive power, what advantage does the lower dimensional representation provide?



```python
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

visualize_prob(logreg_pca, x_train_pca, y_train, ax1)
ax1.set_title('training set')

visualize_prob(logreg_pca, x_test_pca, y_test, ax2)
ax2.set_title('test set')
```





    Text(0.5, 1.0, 'test set')




![png](cs109a_hw5_web_files/cs109a_hw5_web_61_1.png)



<font color='red'>
The spread of probabilities of the PCA model for the training set seems exactly the same as that for multiple logistic regression. In the case of the test set, cancer type 1 probabilities are closer to 1, which can be interpreted as greater certainty for Cancer type 1. Cancer type 0, on the other hand, shows greater spread in the PCA case as compared to the multiple logreg case. There are also cases where Cancer 0 is wrongly classified as Cancer 1, which was absent previously. <br/>

In summary, although the lower dimensional representation yields similar performance as the original multilog model, the computational cost is far lower due the order-of-magnitude difference in the number of features in the PCA-transformed data. It is also worth noting that while the PCA-transformed data has far fewer features than we did originally, this does not necessarily imply that this transformed dataset is more interpretable - the value of each principal component for a datapoint is yielded by a linear combination of the values of all of the original features.
</font>



```python
#--------  visualize_prob

def visualize_prob_2(model1, model2, x1, y1, x2, y2, ax):
    # Use the model to predict probabilities for x
    y1_pred = model1.predict_proba(x1)
    y2_pred = model2.predict_proba(x2)
    
    # Separate the predictions on the label 1 and label 0 points
    ypos1 = y1_pred[y1==1]
    yneg1 = y1_pred[y1==0]
    ypos2 = y2_pred[y2==1]
    yneg2 = y2_pred[y2==0]
    
    # Count the number of label 1 and label 0 points
    npos1 = ypos1.shape[0]
    nneg1 = yneg1.shape[0]
    npos2 = ypos2.shape[0]
    nneg2 = yneg2.shape[0]
    
    
    # Plot the probabilities on a vertical line at x = 0, 
    # with the positive points in blue and negative points in red
    pos_handle = ax.plot(np.zeros((npos1,1))-0.1, ypos1[:,1], 'bo', label = 'MultReg Cancer Type 1')
    neg_handle = ax.plot(np.zeros((nneg1,1))-0.1, yneg1[:,1], 'ro', label = 'MultReg Cancer Type 0')
    pos_handle = ax.plot(np.zeros((npos1,1))+.1, ypos2[:,1], 'go', label = 'PCA Cancer Type 1')
    neg_handle = ax.plot(np.zeros((nneg1,1))+.1, yneg2[:,1], 'mo', label = 'PCA Cancer Type 0')


    # Line to mark prob 0.5
    ax.axhline(y = 0.5, color = 'k', linestyle = '--')
    
    # Add y-label and legend, do not display x-axis, set y-axis limit
    ax.set_ylabel('Probability of AML class')
    ax.legend(loc = 2)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim([0,1])
    ax.set_xlim([-1,1])
```


We also present the above plot juxtaposed with our multireg classification for convenience:



```python
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

visualize_prob_2(multi_regr, logreg_pca, all_x_train, y_train, x_train_pca, y_train, ax1)
ax1.set_title('training set')

visualize_prob_2(multi_regr, logreg_pca, all_x_test, y_test, x_test_pca, y_test, ax2)
ax2.set_title('test set')
plt.show()
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_65_0.png)

