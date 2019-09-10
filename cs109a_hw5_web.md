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

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
```


## Cancer Classification from Gene Expressions 

In this problem, we will build a classification model to distinguish between two related classes of cancer, acute lymphoblastic leukemia (ALL) and acute myeloid leukemia (AML), using gene expression measurements. The data set is provided in the file `data/dataset_hw5_1.csv`. Each row in this file corresponds to a tumor tissue sample from a patient with one of the two forms of Leukemia. The first column contains the cancer type, with 0 indicating the ALL class and 1 indicating the AML class. Columns 2-7130 contain expression levels of 7129 genes recorded from each tissue sample. 

In the following questions, we will use linear and logistic regression to build classification models for this data set. We will also use Principal Components Analysis (PCA) to reduce its dimensions. First step is to split the observations into an approximate 50-50 train-test split. 



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
      <th>AFFX-BioB-5_st</th>
      <th>AFFX-BioB-M_st</th>
      <th>AFFX-BioB-3_st</th>
      <th>AFFX-BioC-5_st</th>
      <th>AFFX-BioC-3_st</th>
      <th>AFFX-BioDn-5_st</th>
      <th>AFFX-BioDn-3_st</th>
      <th>AFFX-CreX-5_st</th>
      <th>AFFX-CreX-3_st</th>
      <th>hum_alu_at</th>
      <th>AFFX-DapX-5_at</th>
      <th>AFFX-DapX-M_at</th>
      <th>AFFX-DapX-3_at</th>
      <th>AFFX-LysX-5_at</th>
      <th>AFFX-LysX-M_at</th>
      <th>AFFX-LysX-3_at</th>
      <th>AFFX-PheX-5_at</th>
      <th>AFFX-PheX-M_at</th>
      <th>AFFX-PheX-3_at</th>
      <th>AFFX-ThrX-5_at</th>
      <th>AFFX-ThrX-M_at</th>
      <th>AFFX-ThrX-3_at</th>
      <th>AFFX-TrpnX-5_at</th>
      <th>AFFX-TrpnX-M_at</th>
      <th>AFFX-TrpnX-3_at</th>
      <th>AFFX-HUMISGF3A/M97935_5_at</th>
      <th>AFFX-HUMISGF3A/M97935_MA_at</th>
      <th>AFFX-HUMISGF3A/M97935_MB_at</th>
      <th>AFFX-HUMISGF3A/M97935_3_at</th>
      <th>AFFX-HUMRGE/M10098_5_at</th>
      <th>AFFX-HUMRGE/M10098_M_at</th>
      <th>AFFX-HUMRGE/M10098_3_at</th>
      <th>AFFX-HUMGAPDH/M33197_5_at</th>
      <th>AFFX-HUMGAPDH/M33197_M_at</th>
      <th>AFFX-HUMGAPDH/M33197_3_at</th>
      <th>AFFX-HSAC07/X00351_5_at</th>
      <th>AFFX-HSAC07/X00351_M_at</th>
      <th>AFFX-HSAC07/X00351_3_at</th>
      <th>AFFX-HUMTFRR/M11507_5_at</th>
      <th>AFFX-HUMTFRR/M11507_M_at</th>
      <th>...</th>
      <th>U22029_f_at</th>
      <th>U49974_f_at</th>
      <th>U65918_f_at</th>
      <th>V00532_rna1_f_at</th>
      <th>V00533_rna1_f_at</th>
      <th>V00542_f_at</th>
      <th>V00551_f_at</th>
      <th>V01516_f_at</th>
      <th>X00090_f_at</th>
      <th>X13930_f_at</th>
      <th>X53065_f_at</th>
      <th>X64177_f_at</th>
      <th>X67491_f_at</th>
      <th>X71345_f_at</th>
      <th>X97444_f_at</th>
      <th>Z80780_f_at</th>
      <th>X00351_f_at</th>
      <th>X01677_f_at</th>
      <th>M31667_f_at</th>
      <th>L41268_f_at</th>
      <th>X99479_f_at</th>
      <th>HG658-HT658_f_at</th>
      <th>M94880_f_at</th>
      <th>S80905_f_at</th>
      <th>X03068_f_at</th>
      <th>Z34822_f_at</th>
      <th>U87593_f_at</th>
      <th>U88902_cds1_f_at</th>
      <th>AC002076_cds2_at</th>
      <th>D64015_at</th>
      <th>HG2510-HT2606_at</th>
      <th>L10717_at</th>
      <th>L34355_at</th>
      <th>L78833_cds4_at</th>
      <th>M13981_at</th>
      <th>M21064_at</th>
      <th>M93143_at</th>
      <th>S78825_at</th>
      <th>U11863_at</th>
      <th>U29175_at</th>
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
      <td>206</td>
      <td>-41</td>
      <td>-831</td>
      <td>-653</td>
      <td>-462</td>
      <td>75</td>
      <td>381</td>
      <td>-118</td>
      <td>-565</td>
      <td>15091</td>
      <td>7</td>
      <td>311</td>
      <td>-231</td>
      <td>21</td>
      <td>-107</td>
      <td>165</td>
      <td>-78</td>
      <td>-204</td>
      <td>29</td>
      <td>-61</td>
      <td>-105</td>
      <td>-366</td>
      <td>-41</td>
      <td>-346</td>
      <td>-297</td>
      <td>-109</td>
      <td>-13</td>
      <td>215</td>
      <td>797</td>
      <td>14538</td>
      <td>9738</td>
      <td>8529</td>
      <td>15076</td>
      <td>11126</td>
      <td>17782</td>
      <td>16287</td>
      <td>18727</td>
      <td>15774</td>
      <td>264</td>
      <td>70</td>
      <td>...</td>
      <td>26</td>
      <td>63</td>
      <td>60</td>
      <td>-20</td>
      <td>-30</td>
      <td>-91</td>
      <td>-43</td>
      <td>488</td>
      <td>1</td>
      <td>504</td>
      <td>391</td>
      <td>-763</td>
      <td>172</td>
      <td>149</td>
      <td>341</td>
      <td>788</td>
      <td>21210</td>
      <td>13771</td>
      <td>598</td>
      <td>396</td>
      <td>245</td>
      <td>14476</td>
      <td>10882</td>
      <td>701</td>
      <td>2762</td>
      <td>-325</td>
      <td>-67</td>
      <td>346</td>
      <td>-68</td>
      <td>229</td>
      <td>-14</td>
      <td>108</td>
      <td>28</td>
      <td>349</td>
      <td>61</td>
      <td>273</td>
      <td>384</td>
      <td>-306</td>
      <td>-1827</td>
      <td>1582</td>
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
      <td>252</td>
      <td>155</td>
      <td>-471</td>
      <td>-490</td>
      <td>-184</td>
      <td>32</td>
      <td>213</td>
      <td>1</td>
      <td>-260</td>
      <td>18128</td>
      <td>-28</td>
      <td>118</td>
      <td>-153</td>
      <td>-8</td>
      <td>-111</td>
      <td>44</td>
      <td>-88</td>
      <td>-102</td>
      <td>32</td>
      <td>5</td>
      <td>-18</td>
      <td>-228</td>
      <td>53</td>
      <td>-348</td>
      <td>-169</td>
      <td>-156</td>
      <td>-55</td>
      <td>122</td>
      <td>483</td>
      <td>1284</td>
      <td>2731</td>
      <td>316</td>
      <td>14653</td>
      <td>15030</td>
      <td>17384</td>
      <td>16386</td>
      <td>19091</td>
      <td>18323</td>
      <td>291</td>
      <td>78</td>
      <td>...</td>
      <td>45</td>
      <td>-29</td>
      <td>23</td>
      <td>-20</td>
      <td>-104</td>
      <td>-47</td>
      <td>-13</td>
      <td>325</td>
      <td>-24</td>
      <td>299</td>
      <td>162</td>
      <td>-56</td>
      <td>279</td>
      <td>183</td>
      <td>259</td>
      <td>605</td>
      <td>18530</td>
      <td>15619</td>
      <td>65</td>
      <td>122</td>
      <td>126</td>
      <td>8443</td>
      <td>8512</td>
      <td>182</td>
      <td>1503</td>
      <td>-78</td>
      <td>29</td>
      <td>159</td>
      <td>18</td>
      <td>71</td>
      <td>42</td>
      <td>44</td>
      <td>-33</td>
      <td>159</td>
      <td>71</td>
      <td>134</td>
      <td>178</td>
      <td>-182</td>
      <td>-179</td>
      <td>626</td>
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
      <td>-66</td>
      <td>208</td>
      <td>-472</td>
      <td>-163</td>
      <td>-192</td>
      <td>-80</td>
      <td>89</td>
      <td>-106</td>
      <td>-251</td>
      <td>27089</td>
      <td>-5</td>
      <td>186</td>
      <td>-18</td>
      <td>-2</td>
      <td>-150</td>
      <td>79</td>
      <td>-57</td>
      <td>-116</td>
      <td>-45</td>
      <td>38</td>
      <td>-79</td>
      <td>-165</td>
      <td>15</td>
      <td>-432</td>
      <td>-231</td>
      <td>-321</td>
      <td>-224</td>
      <td>-17</td>
      <td>152</td>
      <td>8299</td>
      <td>4205</td>
      <td>5352</td>
      <td>19282</td>
      <td>20865</td>
      <td>25787</td>
      <td>11665</td>
      <td>17706</td>
      <td>17559</td>
      <td>237</td>
      <td>14</td>
      <td>...</td>
      <td>-148</td>
      <td>17</td>
      <td>-34</td>
      <td>-10</td>
      <td>27</td>
      <td>-66</td>
      <td>-8</td>
      <td>537</td>
      <td>143</td>
      <td>373</td>
      <td>240</td>
      <td>761</td>
      <td>208</td>
      <td>319</td>
      <td>235</td>
      <td>1434</td>
      <td>14925</td>
      <td>23032</td>
      <td>685</td>
      <td>370</td>
      <td>175</td>
      <td>11348</td>
      <td>10250</td>
      <td>352</td>
      <td>3158</td>
      <td>-271</td>
      <td>-47</td>
      <td>120</td>
      <td>-27</td>
      <td>126</td>
      <td>-24</td>
      <td>2</td>
      <td>-287</td>
      <td>137</td>
      <td>162</td>
      <td>181</td>
      <td>91</td>
      <td>-281</td>
      <td>-1399</td>
      <td>219</td>
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
      <td>173</td>
      <td>-133</td>
      <td>-958</td>
      <td>-271</td>
      <td>-228</td>
      <td>-42</td>
      <td>232</td>
      <td>100</td>
      <td>-599</td>
      <td>14022</td>
      <td>95</td>
      <td>98</td>
      <td>-208</td>
      <td>107</td>
      <td>-237</td>
      <td>84</td>
      <td>-241</td>
      <td>-144</td>
      <td>-29</td>
      <td>22</td>
      <td>-91</td>
      <td>-573</td>
      <td>76</td>
      <td>-732</td>
      <td>-283</td>
      <td>-283</td>
      <td>-226</td>
      <td>265</td>
      <td>808</td>
      <td>-96</td>
      <td>-396</td>
      <td>-382</td>
      <td>16566</td>
      <td>13000</td>
      <td>15983</td>
      <td>14989</td>
      <td>19126</td>
      <td>16599</td>
      <td>489</td>
      <td>166</td>
      <td>...</td>
      <td>157</td>
      <td>318</td>
      <td>36</td>
      <td>28</td>
      <td>-237</td>
      <td>-84</td>
      <td>-54</td>
      <td>625</td>
      <td>142</td>
      <td>1138</td>
      <td>496</td>
      <td>-2227</td>
      <td>655</td>
      <td>399</td>
      <td>476</td>
      <td>784</td>
      <td>22677</td>
      <td>14750</td>
      <td>1334</td>
      <td>698</td>
      <td>305</td>
      <td>10971</td>
      <td>11978</td>
      <td>436</td>
      <td>1116</td>
      <td>-100</td>
      <td>-6</td>
      <td>600</td>
      <td>198</td>
      <td>650</td>
      <td>189</td>
      <td>116</td>
      <td>-414</td>
      <td>659</td>
      <td>40</td>
      <td>299</td>
      <td>897</td>
      <td>-249</td>
      <td>-3381</td>
      <td>917</td>
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
      <td>-55</td>
      <td>-209</td>
      <td>-362</td>
      <td>-427</td>
      <td>-4</td>
      <td>199</td>
      <td>290</td>
      <td>1</td>
      <td>-350</td>
      <td>23507</td>
      <td>41</td>
      <td>93</td>
      <td>-147</td>
      <td>4</td>
      <td>-81</td>
      <td>76</td>
      <td>-67</td>
      <td>-46</td>
      <td>-55</td>
      <td>1</td>
      <td>19</td>
      <td>-267</td>
      <td>-10</td>
      <td>-414</td>
      <td>-133</td>
      <td>-130</td>
      <td>-23</td>
      <td>78</td>
      <td>524</td>
      <td>832</td>
      <td>933</td>
      <td>273</td>
      <td>19483</td>
      <td>16430</td>
      <td>20256</td>
      <td>18222</td>
      <td>23996</td>
      <td>21277</td>
      <td>304</td>
      <td>78</td>
      <td>...</td>
      <td>58</td>
      <td>35</td>
      <td>-15</td>
      <td>-58</td>
      <td>-118</td>
      <td>-67</td>
      <td>-15</td>
      <td>340</td>
      <td>-78</td>
      <td>350</td>
      <td>258</td>
      <td>-252</td>
      <td>73</td>
      <td>273</td>
      <td>377</td>
      <td>-1</td>
      <td>22483</td>
      <td>17487</td>
      <td>1371</td>
      <td>212</td>
      <td>146</td>
      <td>10694</td>
      <td>8760</td>
      <td>494</td>
      <td>8161</td>
      <td>-64</td>
      <td>16</td>
      <td>514</td>
      <td>52</td>
      <td>188</td>
      <td>34</td>
      <td>22</td>
      <td>229</td>
      <td>236</td>
      <td>-62</td>
      <td>172</td>
      <td>376</td>
      <td>-173</td>
      <td>-1066</td>
      <td>453</td>
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




```python
min_vals = data_train.min()
max_vals = data_train.max()

data_train = (data_train - min_vals)/(max_vals - min_vals)
data_test = (data_test - min_vals)/(max_vals - min_vals)
```


**1.2:** Notice that the results training set contains significantly more predictors than observations. Do you foresee a problem in fitting a classification model to such a data set?



```python
data_train.shape
```





    (40, 7130)



The training set is improper as it contains many more columns compared to number of samples. If we fit models to the given dataset, they will be highly overfitted. This is called the curse of dimensionality.

**1.3:** Lets explore a few of the genes and see how well they discriminate between cancer classes. Create a single figure with four subplots arranged in a 2x2 grid. Consider the following four genes: `D29963_at`, `M23161_at`, `hum_alu_at`, and `AFFX-PheX-5_at`. For each gene overlay two histograms of the gene expression values on one of the subplots, one histogram for each cancer type. Does it appear that any of these genes discriminate between the two classes well? How are you able to tell?



```python
alpha=0.5

fig, ax = plt.subplots(2, 2, figsize=(16, 10))

cancer_type_0 = data_train[data_train['Cancer_type'] == 0]
cancer_type_1 = data_train[data_train['Cancer_type'] == 1]

## D29963_at
ax[0, 0].hist(cancer_type_0['D29963_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[0, 0].hist(cancer_type_1['D29963_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[0, 0].set_title('D29963_at')
ax[0, 0].legend()

## M23161_at
ax[0, 1].hist(cancer_type_0['M23161_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[0, 1].hist(cancer_type_1['M23161_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[0, 1].set_title('M23161_at')
ax[0, 1].legend()

## hum_alu_at
ax[1, 0].hist(cancer_type_0['hum_alu_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[1, 0].hist(cancer_type_1['hum_alu_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[1, 0].set_title('hum_alu_at')
ax[1, 0].legend()

## AFFX-PheX-5_at
ax[1, 1].hist(cancer_type_0['AFFX-PheX-5_at'], alpha=alpha, label='cancer type 0', normed=True)
ax[1, 1].hist(cancer_type_1['AFFX-PheX-5_at'], alpha=alpha, label='cancer type 1', normed=True)
ax[1, 1].set_title('AFFX-PheX-5_at')
ax[1, 1].legend();
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_12_0.png)


From the above histograms, we may argue that `D29963_at` and `hum_alu_at` have slightly disjointed regions in the graph and may be singly able to differentiate between the cancer types. For the other two genes, the histograms for both cancer types seem to overlap and don't have distinctive regions. <br/> 

If in any of the above plots, the frequency level of each cancer type was different or the gene measure was in different ranges, we could argue that that a gene is able to distinguish between different cancer types well. This may be said for different frequency levels of `D29963_at` but doesn't seem true in any other case.

It is also useful to note the shape of the individual distributions as well. For instance, `hum_alu_at` has a very bimodal distribution for type 1, whereas for 0 it is more unimodal. The same can be said for `M23161_at` to a lesser extent. Loosely speaking, the characteristics of the distributions could be used to compare the odds of a given gene expression value indicating 0 vs 1. 

Finally, it is worth noting in our solution that `density`/`normed` is not ensuring that the result matches the **true** continuous PDF, but rather simply scaling the counts of the values to turn it into a valid discrete density function determined solely by our samples. This helps us compare the two cancer types, as the number of samples from each cancer type is different. 

**1.4:** Since our data has dimensions that are not easily visualizable, we want to reduce the dimensionality of the data to make it easier to visualize. Using PCA, find the top two principal components for the gene expression data. Generate a scatter plot using these principal components, highlighting the two cancer types in different colors. How well do the top two principal components discriminate between the two classes? How much of the variance within the data do these two principal components explain?



```python
x_train, y_train = data_train.iloc[:,1:], data_train.iloc[:,0]
x_test, y_test = data_test.iloc[:,1:], data_test.iloc[:,0]

pca = PCA(n_components = 2).fit(x_train)
pca_x = pca.transform(x_train)
print("pca shape:", pca_x.shape)
print( "Variance explained: {:.2f}".format(np.sum(pca.explained_variance_ratio_)))

## generating scatter plot
pca_df = pd.DataFrame(pca_x, columns=['PCA1', 'PCA2'])
pca_df['Cancer_type'] = y_train.values

sns.lmplot(x="PCA1", y="PCA2", hue='Cancer_type', data=pca_df, fit_reg=False)
plt.title('Scatter plot of top 2 components', fontsize=14);
```


    pca shape: (40, 2)
    Variance explained: 0.27



![png](cs109a_hw5_web_files/cs109a_hw5_web_15_1.png)


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

print("Number of components that explain at least 90% of the variance: {:.0f}".format(
    sum(np.cumsum(pca.explained_variance_ratio_)<=0.9)+1));
```


    Number of components that explain at least 90% of the variance: 29



![png](cs109a_hw5_web_files/cs109a_hw5_web_18_1.png)


The number of principal components cannot be greater than the number of samples or the number of features. So in this case we can only get at most 40 PCs. As shown above, 29 components are needed to explain at least 90% of the variability in the feature set.

Two components are probably not enough in this context, since they can only explain 27% of the variance in the data. However, the number of components to consider ultimately depends on how we plan to use the transformed data. If we simply want to visualize the classes, then it is clear we cannot use more than three components. If we want to train a classifier on the transformed data (as we do later on), then we could consider a series of different numbers of PCs, and observe how the classifiation accuracy changes based on the number used. We could also consider the point of diminishing returns of adding new PCs, as demonstrated below.



```python
print("The variance explained by first 11 components is {:.2f}".format(np.sum(pca.explained_variance_ratio_[:11])))

plt.scatter(range(1, 41), pca.explained_variance_ratio_)
plt.xlabel('index of components')
plt.ylabel('variance explained')
plt.title('Variance explained by each component', fontsize=15);
```


    The variance explained by first 11 components is 0.60



![png](cs109a_hw5_web_files/cs109a_hw5_web_20_1.png)


From the plot above we see approximately that the first 11 components explain relatively high variance, and the components afterward each explain similar amounts of variance. If we choose the first 11 components based on this observation, we would explain more than 60% of total variance.

## Linear Regression vs. Logistic Regression 


**2.1:** Fit a simple linear regression model to the training set using the single gene predictor `D29963_at` to predict cancer type and plot the histogram of predicted values. We could interpret the scores predicted by the regression model for a patient as an estimate of the probability that the patient has Cancer_type=1 (AML). Is there a problem with this interpretation?



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

## You can specify a rotation for the tick labels in degrees or with keywords.
par1.set_yticks( [0.082, 0.81]);
par1.set_yticklabels(labels);
```


    shapes of x_train and x_test (40, 2) (33, 2)



![png](cs109a_hw5_web_files/cs109a_hw5_web_24_1.png)




```python
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
axs.hist(y_train_pred, alpha=0.5)
axs.set_xlabel("Response Values of Trained Model")
axs.set_ylabel("Count")
axs.set_title("Output Probability Histogram on Training Data")
plt.show()
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_25_0.png)


If we take the regression prediction as probability of having Cancer Type 1, it is not clear how we will determine the presence of having cancer type 1 - is it 1-probability of Cancer type 1? We will also not be able to interpret response values $> 1$ or $< 0$.

**2.2:** The fitted linear regression model can be converted to a classification model (i.e. a model that predicts one of two binary classes 0 or 1) by classifying patients with predicted score greater than 0.5 into `Cancer_type`=1, and the others into the `Cancer_type`=0. Evaluate the classification accuracy of the obtained classification model on both the training and test sets.



```python
train_score = accuracy_score(y_train, y_train_pred>0.5)
test_score = accuracy_score(y_test, y_test_pred>0.5)
print("train score: {0:.2f}, test score: {1:.2f}".format(train_score, test_score))
```


    train score: 0.80, test score: 0.76


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

print("train score: {0:.2f}, test score: {1:.2f}".format(train_score_logreg, test_score_logreg))
```


    train score: 0.80, test score: 0.76


Logistic regression gave the same accuracy as Linear Regression model. Classifications are going to always be similar (especially in simple models) with few points. 

**2.4:** Create a figure with 4 items displayed on the same plot:
- the quantitative response from the linear regression model as a function of the gene predictor `D29963_at`.
- the predicted probabilities of the logistic regression model as a function of the gene predictor `D29963_at`.  
- the true binary response for the test set points for both models in the same plot. 
- a horizontal line at $y=0.5$. 

Based on these plots, does one of the models appear better suited for binary classification than the other?  Explain in 3 sentences or fewer. 




```python
plt.figure(figsize=(12,6))
sort_index = np.argsort(x_test['D29963_at'].values)

## plotting true binary response
plt.scatter(x_test['D29963_at'].iloc[sort_index], y_test.iloc[sort_index], color='black', label = 'True Response')

## plotting ols output
plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred.iloc[sort_index], color='red', alpha=0.3, \
         label = 'Linear Regression Predictions')

## plotting logreg prob output
plt.plot(x_test['D29963_at'].iloc[sort_index], y_test_pred_logreg_prob[sort_index], alpha=0.3,  \
         color='green', label = 'Logistic Regression Predictions Prob')

plt.axhline(0.5, c='c')
plt.legend()
plt.title('True response v/s obtained responses')
plt.xlabel('Gene predictor value')
plt.ylabel('Cancer type response');
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_33_0.png)


Both models give similar outputs as can be observed from the plot above. However, we can see that Linear regression output goes below 0 and above 1 in some cases, whereas Logistic regression outputs are probability values between 0 and 1.

## Multiple Logistic Regression 



```python
## --------  visualize_prob
## A function to visualize the probabilities predicted by a Logistic Regression model
## Input: 
##      model (Logistic regression model)
##      x (n x d array of predictors in training data)
##      y (n x 1 array of response variable vals in training data: 0 or 1)
##      ax (an axis object to generate the plot)

def visualize_prob(model, x, y, ax):
    ## Use the model to predict probabilities for x
    y_pred = model.predict_proba(x)
    
    ## Separate the predictions on the label 1 and label 0 points
    ypos = y_pred[y==1]
    yneg = y_pred[y==0]
    
    ## Count the number of label 1 and label 0 points
    npos = ypos.shape[0]
    nneg = yneg.shape[0]
    
    ## Plot the probabilities on a vertical line at x = 0, 
    ## with the positive points in blue and negative points in red
    pos_handle = ax.plot(np.zeros((npos,1)), ypos[:,1], 'bo', label = 'Cancer Type 1')
    neg_handle = ax.plot(np.zeros((nneg,1)), yneg[:,1], 'ro', label = 'Cancer Type 0')

    ## Line to mark prob 0.5
    ax.axhline(y = 0.5, color = 'k', linestyle = '--')
    
    ## Add y-label and legend, do not display x-axis, set y-axis limit
    ax.set_ylabel('Probability of AML class')
    ax.legend(loc = 'best')
    ax.get_xaxis().set_visible(False)
    ax.set_ylim([0,1])
```


**3.1:** Next, fit a multiple logistic regression model with all the gene predictors from the data set. How does the classification accuracy of this model compare with the models fitted in question 2 with a single gene (on both the training and test sets)?



```python
all_x_train = x_train 
all_x_test = x_test 

## fitting multi regression model
multi_regr = LogisticRegression(C=100000, solver = "lbfgs")
multi_regr.fit(all_x_train, y_train)

y_train_pred_multi = multi_regr.predict(all_x_train) 
y_test_pred_multi = multi_regr.predict(all_x_test)

train_score_multi = accuracy_score(y_train, y_train_pred_multi)
test_score_multi = accuracy_score(y_test, y_test_pred_multi)

print('Training set accuracy for multiple logistic regression: {:.2f}'.format(train_score_multi))
print('Test set accuracy for multiple logistic regression: {:.2f}'.format(test_score_multi))
```


    Training set accuracy for multiple logistic regression: 1.00
    Test set accuracy for multiple logistic regression: 1.00


The multiple logistic regression accuracy for both train and test sets are perfect at 100%. This is an improvement from the Logistic regression accuracy with single predictors. However, the moral of the story here is that (1) because of the shortage of training samples, we are overfitting to noise in the training, and (2) because of the shortage of test samples and the high dimensionality of the data, we have no way of ascertaining the generalization capacity of any trained model - which could mislead us into a false sense of security on the accuracy of the model as a whole. 

<font color='red'>
    
**3.2** How many of the coefficients estimated by this multiple logistic regression in the previous part are significantly different from zero at a *significance level of 5%*? Use the same value of C=100000 as before.

**Hint:** To answer this question, use *bootstrapping* with 1000 boostrap samples/iterations.  
</font>



```python
## bootstrapping code
n = 1000 
boot_coefs = np.zeros((all_x_train.shape[1],n)) # Create empty storage array for later use

## iteration for each sample
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


    Significant coefficents at 5pct level = 1878 / 7129


Thus, we can see that only 1878 out of 7130 predictors are significantly different from 0 at a significance level of 5%.

**3.3** Use the `visualize_prob` function provided below (or any other visualization) to visualize the probabilties predicted by the fitted multiple logistic regression model on both the training and test data sets. The function creates a visualization that places the data points on a vertical line based on the predicted probabilities, with the different cancer classes shown in different colors, and with the 0.5 threshold highlighted using a dotted horizontal line. Is there a difference in the spread of probabilities in the training and test plots? Are there data points for which the predicted probability is close to 0.5? If so, what can you say about these points?



```python
""" Plot classification model """
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

## visualising train set fit
visualize_prob(multi_regr, all_x_train, y_train, ax1)
ax1.set_title('training set')

## visualising test set fit
visualize_prob(multi_regr, all_x_test, y_test, ax2)
ax2.set_title('test set');
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_45_0.png)


Difference in spread of probability - In the training set the predicted AML values are 1 and predicted ALL values are 0. In the test set, the probabilities are spread above 0.5 but closer to 0 for points below 0.5. There are data points for which probability is close to 0.5, indicating that it cannot be determined clearly which type of cancer the patient has. They may give false positives or false negatives.


**3.4** Given the results above how would you assest the generalization capacity of your trained model? What other tests would you suggest to better guard against false sense of security on the accuracy of the model as a whole. 

**Answer:** Cross validation 

## PCR: Principal Components Regression

High dimensional problems can lead to problematic behavior in model estimation (and make prediction on a test set worse), thus we often want to try to reduce the dimensionality of our problems. A reasonable approach to reduce the dimensionality of the data is to use PCA and fit a logistic regression model on the smallest set of principal components that explain at least 90% of the variance in the predictors.

**4.1** Fit two separate Logistic Regression models using principal components as the predictors: (1) with the number of components you selected from problem 1.5 and (2) with the number of components that explain at least 90% of the variability in the feature set. How do the classification accuracy values on both the training and tests sets compare with the models fit in question 3?



```python
## Applying PCA 
pca = PCA(n_components=29) # from our result in 1.5
pca.fit(x_train)

## transforming train and test data for regression
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

logreg_pca = LogisticRegression(C=100000, fit_intercept=False)
logreg_pca.fit(x_train_pca, y_train)

y_train_pred_logreg_pca = logreg_pca.predict(x_train_pca)
y_test_pred_logreg_pca = logreg_pca.predict(x_test_pca)

train_score_logreg_pca = accuracy_score(y_train, y_train_pred_logreg_pca)
test_score_logreg_pca = accuracy_score(y_test, y_test_pred_logreg_pca)

print('Training set accuracy for Logistic Regression with PCA = {:.2f}'.format(train_score_logreg_pca))
print('Test set accuracy for Logistic Regression with PCA = {:.2f}'.format(test_score_logreg_pca))
```


    Training set accuracy for Logistic Regression with PCA = 1.00
    Test set accuracy for Logistic Regression with PCA = 0.94


1. The first 20 principal components explain 90.26% of the total variance in data. <br/>
2. Logistic Regression on PCA components gives 100% train accuracy and 96.96% test accuracy. The train accuracy here is same as LogReg in 3.1 but test accuracy is slightly less tha Logreg. Since PCA components only explain 90% of the data, this makes intuitive sense. <br/>

**4.2:** Use the code provided in question 3 to visualize the probabilities predicted by the fitted models on both the training and test sets. How does the spread of probabilities in these plots compare to those for the model in question 3.2? If the lower dimensional representation yields comparable predictive power, what advantage does the lower dimensional representation provide?



```python
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

## visualising train set fit
visualize_prob(logreg_pca, x_train_pca, y_train, ax1)
ax1.set_title('training set')

## visualising test set fit
visualize_prob(logreg_pca, x_test_pca, y_test, ax2)
ax2.set_title('test set');
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_53_0.png)


The spread of probabilities of the PCA model for the training set seems exactly the same as that for multiple logistic regression. In the case of the test set, cancer type 1 probabilities are closer to 1, which can be interpreted as greater certainty for Cancer type 1. Cancer type 0, on the other hand, shows greater spread in the PCA case as compared to the multiple logreg case. There are also cases where Cancer 0 is wrongly classified as Cancer 1, which was absent previously. <br/>

In summary, although the lower dimensional representation yields similar performance as the original multilog model, the computational cost is far lower due the order-of-magnitude difference in the number of features in the PCA-transformed data. It is also worth noting that while the PCA-transformed data has far fewer features than we did originally, this does not necessarily imply that this transformed dataset is more interpretable - the value of each principal component for a datapoint is yielded by a linear combination of the values of all of the original features.



```python
## --------  visualize_prob
## A function to visualize the probabilities predicted by a Logistic Regression model
## Input: 
##      model (Logistic regression model)
##      x (n x d array of predictors in training data)
##      y (n x 1 array of response variable vals in training data: 0 or 1)
##      ax (an axis object to generate the plot)

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

## visualising train set fit
visualize_prob_2(multi_regr, logreg_pca, all_x_train, y_train, x_train_pca, y_train, ax1)
ax1.set_title('training set')

## visualising test set fit
visualize_prob_2(multi_regr, logreg_pca, all_x_test, y_test, x_test_pca, y_test, ax2)
ax2.set_title('test set')
plt.show()
```



![png](cs109a_hw5_web_files/cs109a_hw5_web_57_0.png)

