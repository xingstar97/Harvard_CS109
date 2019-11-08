---
title: H16
notebook: cs109b_hw7_web.ipynb
nav_include: 17
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

import requests
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.io import imread
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Reshape, \
UpSampling2D, InputLayer, Lambda, ZeroPadding2D, Cropping2D, Conv2DTranspose, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.losses import binary_crossentropy
from keras import backend as K,objectives
from keras.losses import mse, binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import random

import glob
import skimage
import skimage.transform
import skimage.io
import cv2

from keras_preprocessing.image import ImageDataGenerator
from keras.layers import *
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output

from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
```


    Using TensorFlow backend.


## Generative Adversarial Network

We'll be using a subset of the Celeb A dataset to help us build facial generative models, as described on the [Celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) website. CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. 

CelebA has large diversities, large quantities, and rich annotations, including

- 10,177 number of identities,
- 202,599 number of face images, 
- 5 landmark locations, and 
- 40 binary attributes annotations per image.

### Preparing the Data

**1.A.1.** Download the dataset from: [CelebA Dataset](https://s3.amazonaws.com/gec-harvard-dl2-hw2-data/datasets/celeba-dataset.zip). In the provided data you'll see the `img_align_celeba` directory. The images in that directory will form your dataset.  You may want to create a [Keras Custom Data Generator](https://techblog.appnexus.com/a-keras-multithreaded-dataframe-generator-for-millions-of-image-files-84d3027f6f43).  



```python
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, files, batch_size, target_height, target_width, conditioning_dim=0, conditioning_data=None):
        '''
        Intializes the custom generator.
        
        Args:
            files: The list of paths to images that should be fed to the network.
            batch_size: The batchsize to use.
            target_height: The target image height. If different, the images will be resized.
            target_width: The target image width. If different, the images will be resized.
            conditioning_dim: The dimension of the conditional variable space. Can be 0.
            conditioning_data: Optional dictionary that maps from the filename to the data to be
                conditioned on. Data must be numeric. Can be None. Otherwise, len must be equal to
                conditioning_dim.
        '''
        self.files = files
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.conditioning_dim = conditioning_dim
        self.conditioning_data = conditioning_data

    def on_epoch_end(self):
        '''Shuffle list of files after each epoch.'''
        np.random.shuffle(self.files)
        
    def __getitem__(self, index):
        cur_files = self.files[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(cur_files)
        return X, y
    
    def __data_generation(self, cur_files):
        X = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        Y = np.empty(shape=(self.batch_size, self.target_height, self.target_width, 3))
        if self.conditioning_data != None:
            C = np.empty(shape=(self.batch_size, self.conditioning_dim))
        
        for i, file in enumerate(cur_files):
            img = skimage.io.imread(file)
            if img.shape[0] != self.target_height or img.shape[1] != self.target_width:
                img = skimage.transform.resize(img, (self.target_height, self.target_width)) # Resize.
            img = img.astype(np.float32) 
            X[i] = img
            Y[i] = img
            if self.conditioning_data != None:
                C[i] = self.conditioning_data[os.path.basename(file)]
                
        if self.conditioning_data != None:
            return [X, C], Y
        else:
            return X, Y
    
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
```




```python
files = glob.glob('data/img_align_celeba/*.jpg')
print(len(files), 'images found.')
```


    202599 images found.




```python
VARIATIONAL = True
HEIGHT = 128
WIDTH = 128
BATCH_SIZE = 32
LATENT_DIM = 16
START_FILTERS = 32
```




```python
gen = CustomDataGenerator(files=files, 
                          batch_size=BATCH_SIZE, 
                          target_height=HEIGHT, 
                          target_width=WIDTH)
```


**1.A.2.** Load the attribute data in `list_attr_celeba.csv` into a pandas dataframe.



```python
df = pd.read_csv('data/list_attr_celeba.csv')
df.head()
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
      <th>image_id</th>
      <th>5_o_Clock_Shadow</th>
      <th>Arched_Eyebrows</th>
      <th>Attractive</th>
      <th>Bags_Under_Eyes</th>
      <th>Bald</th>
      <th>Bangs</th>
      <th>Big_Lips</th>
      <th>Big_Nose</th>
      <th>Black_Hair</th>
      <th>Blond_Hair</th>
      <th>Blurry</th>
      <th>Brown_Hair</th>
      <th>Bushy_Eyebrows</th>
      <th>Chubby</th>
      <th>Double_Chin</th>
      <th>Eyeglasses</th>
      <th>Goatee</th>
      <th>Gray_Hair</th>
      <th>Heavy_Makeup</th>
      <th>High_Cheekbones</th>
      <th>Male</th>
      <th>Mouth_Slightly_Open</th>
      <th>Mustache</th>
      <th>Narrow_Eyes</th>
      <th>No_Beard</th>
      <th>Oval_Face</th>
      <th>Pale_Skin</th>
      <th>Pointy_Nose</th>
      <th>Receding_Hairline</th>
      <th>Rosy_Cheeks</th>
      <th>Sideburns</th>
      <th>Smiling</th>
      <th>Straight_Hair</th>
      <th>Wavy_Hair</th>
      <th>Wearing_Earrings</th>
      <th>Wearing_Hat</th>
      <th>Wearing_Lipstick</th>
      <th>Wearing_Necklace</th>
      <th>Wearing_Necktie</th>
      <th>Young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001.jpg</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002.jpg</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000003.jpg</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000004.jpg</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000005.jpg</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**1.A.3.** Pick 3 random images from the dataset, plot them, and verify that the attributes are accurate.



```python
rnd_file = np.random.choice(files)
file_id = os.path.basename(rnd_file)
img = skimage.io.imread(rnd_file)
plt.imshow(img)
plt.show()

init_meta = df[df.image_id==file_id]
display(init_meta)
```



![png](cs109b_hw7_web_files/cs109b_hw7_web_12_0.png)



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
      <th>image_id</th>
      <th>5_o_Clock_Shadow</th>
      <th>Arched_Eyebrows</th>
      <th>Attractive</th>
      <th>Bags_Under_Eyes</th>
      <th>Bald</th>
      <th>Bangs</th>
      <th>Big_Lips</th>
      <th>Big_Nose</th>
      <th>Black_Hair</th>
      <th>Blond_Hair</th>
      <th>Blurry</th>
      <th>Brown_Hair</th>
      <th>Bushy_Eyebrows</th>
      <th>Chubby</th>
      <th>Double_Chin</th>
      <th>Eyeglasses</th>
      <th>Goatee</th>
      <th>Gray_Hair</th>
      <th>Heavy_Makeup</th>
      <th>High_Cheekbones</th>
      <th>Male</th>
      <th>Mouth_Slightly_Open</th>
      <th>Mustache</th>
      <th>Narrow_Eyes</th>
      <th>No_Beard</th>
      <th>Oval_Face</th>
      <th>Pale_Skin</th>
      <th>Pointy_Nose</th>
      <th>Receding_Hairline</th>
      <th>Rosy_Cheeks</th>
      <th>Sideburns</th>
      <th>Smiling</th>
      <th>Straight_Hair</th>
      <th>Wavy_Hair</th>
      <th>Wearing_Earrings</th>
      <th>Wearing_Hat</th>
      <th>Wearing_Lipstick</th>
      <th>Wearing_Necklace</th>
      <th>Wearing_Necktie</th>
      <th>Young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>120528</th>
      <td>120529.jpg</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
rnd_file = np.random.choice(files)
file_id = os.path.basename(rnd_file)
img = skimage.io.imread(rnd_file)
plt.imshow(img)
plt.show()

init_meta = df[df.image_id==file_id]
display(init_meta)
```



![png](cs109b_hw7_web_files/cs109b_hw7_web_13_0.png)



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
      <th>image_id</th>
      <th>5_o_Clock_Shadow</th>
      <th>Arched_Eyebrows</th>
      <th>Attractive</th>
      <th>Bags_Under_Eyes</th>
      <th>Bald</th>
      <th>Bangs</th>
      <th>Big_Lips</th>
      <th>Big_Nose</th>
      <th>Black_Hair</th>
      <th>Blond_Hair</th>
      <th>Blurry</th>
      <th>Brown_Hair</th>
      <th>Bushy_Eyebrows</th>
      <th>Chubby</th>
      <th>Double_Chin</th>
      <th>Eyeglasses</th>
      <th>Goatee</th>
      <th>Gray_Hair</th>
      <th>Heavy_Makeup</th>
      <th>High_Cheekbones</th>
      <th>Male</th>
      <th>Mouth_Slightly_Open</th>
      <th>Mustache</th>
      <th>Narrow_Eyes</th>
      <th>No_Beard</th>
      <th>Oval_Face</th>
      <th>Pale_Skin</th>
      <th>Pointy_Nose</th>
      <th>Receding_Hairline</th>
      <th>Rosy_Cheeks</th>
      <th>Sideburns</th>
      <th>Smiling</th>
      <th>Straight_Hair</th>
      <th>Wavy_Hair</th>
      <th>Wearing_Earrings</th>
      <th>Wearing_Hat</th>
      <th>Wearing_Lipstick</th>
      <th>Wearing_Necklace</th>
      <th>Wearing_Necktie</th>
      <th>Young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>118082</th>
      <td>118083.jpg</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
rnd_file = np.random.choice(files)
file_id = os.path.basename(rnd_file)
img = skimage.io.imread(rnd_file)
plt.imshow(img)
plt.show()

init_meta = df[df.image_id==file_id]
display(init_meta)
```



![png](cs109b_hw7_web_files/cs109b_hw7_web_14_0.png)



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
      <th>image_id</th>
      <th>5_o_Clock_Shadow</th>
      <th>Arched_Eyebrows</th>
      <th>Attractive</th>
      <th>Bags_Under_Eyes</th>
      <th>Bald</th>
      <th>Bangs</th>
      <th>Big_Lips</th>
      <th>Big_Nose</th>
      <th>Black_Hair</th>
      <th>Blond_Hair</th>
      <th>Blurry</th>
      <th>Brown_Hair</th>
      <th>Bushy_Eyebrows</th>
      <th>Chubby</th>
      <th>Double_Chin</th>
      <th>Eyeglasses</th>
      <th>Goatee</th>
      <th>Gray_Hair</th>
      <th>Heavy_Makeup</th>
      <th>High_Cheekbones</th>
      <th>Male</th>
      <th>Mouth_Slightly_Open</th>
      <th>Mustache</th>
      <th>Narrow_Eyes</th>
      <th>No_Beard</th>
      <th>Oval_Face</th>
      <th>Pale_Skin</th>
      <th>Pointy_Nose</th>
      <th>Receding_Hairline</th>
      <th>Rosy_Cheeks</th>
      <th>Sideburns</th>
      <th>Smiling</th>
      <th>Straight_Hair</th>
      <th>Wavy_Hair</th>
      <th>Wearing_Earrings</th>
      <th>Wearing_Hat</th>
      <th>Wearing_Lipstick</th>
      <th>Wearing_Necklace</th>
      <th>Wearing_Necktie</th>
      <th>Young</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>91985</th>
      <td>091986.jpg</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


I think most of the attributes are accurate, even though some attributes are quite subjective, like attractive. 

### Variational Autoencoder Model 

**1.B.1.** Create and compile a Convolutional Variational Autoencoder Model (including encoder and decoder) for the celebrity faces dataset.  Print summaries for the encoder, decoder and full autoencoder models.



```python
def define_encoder_block(x, num_filters):  
    """
    Todo: Define two sequential 2D convolutional layers (Conv2D) with the following properties:
          - num_filters many filters
          - kernel_size 3
          - activation "relu"
          - padding "same"
          - kernel_initializer "he_normal"
          Also define a 2D max pooling layer (MaxPooling2D) (you can keep default arguments).
    """
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D()(x)
    return x
```




```python
def define_decoder_block(x, num_filters):
    """
    Todo: Define one 2D upsampling layer (UpSampling2D) (you can keep default arguments).
          Also, define two sequential 2D convolutional layers (Conv2D) with the following properties:
          - num_filters many filters
          - kernel_size 3
          - activation "relu"
          - padding "same"
          - kernel_initializer "he_normal"
    """
    x = UpSampling2D()(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(num_filters, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    return x
```




```python
def define_net(variational, height, width, batch_size, latent_dim, conditioning_dim=0,
               start_filters=8):
    """Defines a (variational) encoder-decoder architecture.
    
    Args:
        variational: Whether a variational autoencoder should be defined.
        height: The height of the image input and output.
        width: The width of the image input and output.
        batch_size: The batchsize that is used during training. Must also be used for inference on the encoder side.
        latent_dim: The dimension of the latent space.
        conditioning_dim: The dimension of the space of variables to condition on. Can be zero for an 
            unconditional VAE.
        start_filters: The number of filters to start from. Multiples of this value are used across the network. 
            Can be used to change model capacity.
        
    Returns:
        Tuple of keras models for full VAE, encoder part and decoder part only.
    """
    
    # Prepare the inputs.
    inputs = Input((height, width, 3))
    if conditioning_dim > 0:
        # Define conditional VAE. Note that this is usually not the preferred way
        # of incorporating the conditioning information in the encoder.
        condition = Input([conditioning_dim])
        condition_up = Dense(height * width)(condition)
        condition_up = Reshape([height, width, 1])(condition_up)
        inputs_new = Concatenate(axis=3)([inputs, condition_up])
    else:
        inputs_new = inputs
        
    # Define the encoder.
    eblock1 = define_encoder_block(inputs_new, start_filters)
    eblock2 = define_encoder_block(eblock1, start_filters*2)
    eblock3 = define_encoder_block(eblock2, start_filters*4)
    eblock4 = define_encoder_block(eblock3, start_filters*8)
    _, *shape_spatial = eblock4.get_shape().as_list()
    eblock4_flat = Flatten()(eblock4)
    
    if not variational:
        z = Dense(latent_dim)(eblock4_flat)
    else:
        # Perform the sampling.
        def sampling(args):
            """Samples latent variable from a normal distribution using the given parameters."""
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon
        
        z_mean = Dense(latent_dim)(eblock4_flat)
        z_log_sigma = Dense(latent_dim)(eblock4_flat)
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    if conditioning_dim > 0:
        z_ext = Concatenate()([z, condition])

    # Define the decoder.
    inputs_embedding = Input([latent_dim + conditioning_dim])
    embedding = Dense(np.prod(shape_spatial), activation='relu')(inputs_embedding)
    embedding = Reshape(eblock4.shape.as_list()[1:])(embedding)
    
    dblock1 = define_decoder_block(embedding, start_filters*8)
    dblock2 = define_decoder_block(dblock1, start_filters*4)
    dblock3 = define_decoder_block(dblock2, start_filters*2)
    dblock4 = define_decoder_block(dblock3, start_filters)
    output = Conv2D(3, 1, activation = 'tanh')(dblock4)
    
    # Define the models.
    decoder = Model(input = inputs_embedding, output = output)
    if conditioning_dim > 0:
        encoder_with_sampling = Model(input = [inputs, condition], output = z)
        encoder_with_sampling_ext = Model(input = [inputs, condition], output = z_ext)
        vae_out = decoder(encoder_with_sampling_ext([inputs, condition]))
        vae = Model(input = [inputs, condition], output = vae_out)
    else:
        encoder_with_sampling = Model(input = inputs, output = z)
        vae_out = decoder(encoder_with_sampling(inputs))
        vae = Model(input = inputs, output = vae_out)
    
    # Define the VAE loss.
    def vae_loss(x, x_decoded_mean):
        """Defines the VAE loss functions as a combination of MSE and KL-divergence loss."""
        mse_loss = K.mean(keras.losses.mse(x, x_decoded_mean), axis=(1,2)) * height * width
        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1) # -5e-4
        return mse_loss + kl_loss
    
    if variational:
        vae.compile(loss=vae_loss, optimizer='adam')
    else:
        vae.compile(loss='mse', optimizer='adam')    
    
    print('done,', vae.count_params(), 'parameters.')
    
    return vae, encoder_with_sampling, decoder
```




```python
def encode_image(img, conditioning, encoder, height, width, batch_size):
    '''Encodes an image that is given in RGB-channel order with value range of [0, 255].
    
    Args:
        img: The image input. If shapes differ from (height, width), it will be resized.
        conditoning: The set of values to condition on, if any. Can be None.
        encoder: The keras encoder model to use.
        height: The target image height.
        width: The target image width.
        batch_size: The batchsize that the encoder expects.
        
    Returns:
        The latent representation of the input image.
    '''
    if img.shape[0] != height or img.shape[1] != width:
        img = skimage.transform.resize(img, (height, width))
    img_single = np.expand_dims(img, axis=0)
    img_single = img_single.astype(np.float32)
    img_single = np.repeat(img_single, batch_size, axis=0)
    if conditioning is None:
        z = encoder.predict(img_single)
    else:
        z = encoder.predict([img_single, np.repeat(np.expand_dims(conditioning, axis=0), batch_size, axis=0)])
    return z
```




```python
def decode_embedding(z, conditioning, decoder):
    '''Decodes the given representation into an image.
    
    Args:
        z: The latent representation.
        conditioning: The set of values to condition on, if any. Can be None.
        decoder: The keras decoder model to use.
    '''
    if z.ndim < 2:
        z = np.expand_dims(z, axis=0) # Single-batch
    if conditioning is not None:
        z = np.concatenate((z, np.repeat(np.expand_dims(conditioning, axis=0), z.shape[0], axis=0)), axis=1)
    return decoder.predict(z)
```




```python
def load_weights(folder):
    vae.load_weights(folder + '/vae.w')
    encoder.load_weights(folder + '/encoder.w')
    decoder.load_weights(folder + '/decoder.w')
    
def save_weights(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    vae.save_weights(folder + '/vae.w')
    encoder.save_weights(folder + '/encoder.w')
    decoder.save_weights(folder + '/decoder.w')
```




```python
vae, encoder, decoder = define_net(variational=VARIATIONAL,
                                   height=HEIGHT, 
                                   width=WIDTH, 
                                   batch_size=BATCH_SIZE, 
                                   latent_dim=LATENT_DIM,
                                   start_filters=START_FILTERS)
```


    done, 3736419 parameters.




```python
encoder.summary()
```


    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, 128, 128, 3)  0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 128, 128, 32) 896         input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 128, 128, 32) 9248        conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 32)   0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 64, 64, 64)   18496       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 64, 64, 64)   36928       conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 64)   0           conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 32, 32, 128)  73856       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 32, 32, 128)  147584      conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 128)  0           conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 16, 16, 256)  295168      max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 16, 16, 256)  590080      conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 256)    0           conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 16384)        0           max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 16)           262160      flatten_1[0][0]                  
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 16)           262160      flatten_1[0][0]                  
    __________________________________________________________________________________________________
    lambda_1 (Lambda)               (None, 16)           0           dense_1[0][0]                    
                                                                     dense_2[0][0]                    
    ==================================================================================================
    Total params: 1,696,576
    Trainable params: 1,696,576
    Non-trainable params: 0
    __________________________________________________________________________________________________




```python
decoder.summary()
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 16)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 16384)             278528    
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 8, 8, 256)         0         
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 16, 16, 256)       590080    
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 16, 16, 256)       590080    
    _________________________________________________________________
    up_sampling2d_2 (UpSampling2 (None, 32, 32, 256)       0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 32, 32, 128)       295040    
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 32, 32, 128)       147584    
    _________________________________________________________________
    up_sampling2d_3 (UpSampling2 (None, 64, 64, 128)       0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 64, 64, 64)        73792     
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 64, 64, 64)        36928     
    _________________________________________________________________
    up_sampling2d_4 (UpSampling2 (None, 128, 128, 64)      0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 128, 128, 32)      18464     
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 128, 128, 32)      9248      
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 128, 128, 3)       99        
    =================================================================
    Total params: 2,039,843
    Trainable params: 2,039,843
    Non-trainable params: 0
    _________________________________________________________________




```python
vae.summary()
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 128, 128, 3)       0         
    _________________________________________________________________
    model_2 (Model)              (None, 16)                1696576   
    _________________________________________________________________
    model_1 (Model)              (None, 128, 128, 3)       2039843   
    =================================================================
    Total params: 3,736,419
    Trainable params: 3,736,419
    Non-trainable params: 0
    _________________________________________________________________


**1.B.2.** Train your model on the images in the celebA dataset.



```python
```




```python
load_weights('models/celeba_vae_full')
```


**1.B.3.** Choose a random input image.  Encode the image and then Decode the latent representation.  Plot the original image and the reconstructed output.  How do they compare?



```python
np.random.seed(6)
for _ in range(4):
    rnd_file = np.random.choice(files)
    file_id = os.path.basename(rnd_file)
    init_meta = df[df.image_id==file_id]
    init_meta = list(init_meta.values[0][1:])
    img = skimage.io.imread(rnd_file)

    # Encode the image
    z = encode_image(img.astype(np.float32)/255., None, encoder, HEIGHT, WIDTH, BATCH_SIZE)
    print('latent sample:\n', z[0])

    # Decode the image
    ret = decode_embedding(z, None, decoder)

    # Compare the two plots
    f, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax[0].imshow(img)
    ax[1].imshow(ret[0])
    plt.show()
```


    latent sample:
     [-1.453647   -1.163708   -0.2652519   0.6670451   0.2861778  -0.07194141
     -0.0878744   0.6605689   0.24025117 -0.23390593  1.0922871   0.18089256
      0.9380422  -0.15924565 -0.11851126  0.33846545]



![png](cs109b_hw7_web_files/cs109b_hw7_web_32_1.png)


    latent sample:
     [-0.63976693 -0.14634675 -0.44023338  0.02930867 -0.19757304 -0.29004806
     -0.11635195 -0.13232085 -0.3452685   0.7949826  -1.8409612   0.46917525
      0.05666112  0.20296499  0.3155882  -0.15347195]



![png](cs109b_hw7_web_files/cs109b_hw7_web_32_3.png)


    latent sample:
     [ 0.5975285   0.04404822 -2.187132   -0.17378211  0.22034664 -0.05780643
     -0.2060868   0.18818219  0.5698013   0.22895676  2.3336518   0.20302285
     -0.00461983 -0.01116885 -0.17222509  1.6181383 ]



![png](cs109b_hw7_web_files/cs109b_hw7_web_32_5.png)


    latent sample:
     [ 0.68904877 -0.43262032 -0.16549951 -0.18405214 -0.24844773  1.1270659
     -0.11586931 -0.48150706  0.02075912  1.4364051  -0.1878098  -0.12392135
      0.3314267   0.3542386   0.10886622 -0.41668698]



![png](cs109b_hw7_web_files/cs109b_hw7_web_32_7.png)


I compared four images instead of one, and I found they are reasonably close. The generated pictures have all the key features kinda similar to the original pictures. 

**1.B.4.** Choose two celebrity faces from the dataset that differ according to two attributes and taking advantages of alterations of the latent representations image morph from one to the other.  See below for an example.

![](latent_1.png)



```python
def display_manifold(decoder, height, width, base_vec, bound_x=15, bound_y=15, axis_x=0, axis_y=1, n=15,
                     desc_x = 'x', desc_y = 'y', file_out=None):
    '''Varies up to two dimensions of the latent representation, and visualizes its effect.

    This function can be used in one or two dimensions. To just vary a single dimension, set
    either bound_x or bound_y to zero.

    Args:
        decoder: The keras decoder model to use.
        height: The image height the decoder produces.
        width: The image width the decoder produces.
        base_vec: The basic latent representation to which changes should be applied.
            Per convention, the first entries in base_vec correspond to the latent variables,
            followed by variables we condition on (if any). Therefore, dimension is the sum of
            the latent dimension and the conditioning dimension.
        bound_x: The range that the values on axis_x will be modified to.
        bound_y: The range that the values on axis_y will be modified to.
        axis_x: The first axis to modify. Must be 0 <= axis_x <= len(base_vec).
        axis_y: The first axis to modify. Must be 0 <= axis_y <= len(base_vec).
        n: The number of columns/rows to generate. Thus, in total, n**2 images will be generated
            if two dimensions are modified. Otherwise, just n images will be generated.
        desc_x: The caption of the x-axis shown on the plot.
        desc_y: The caption of the y-axis shown on the plot.
        file_out: File path if the resulting plot should be saved. Can be None.

    Returns:
        Results will be plotted. In addition, a tuple is returned, containing both the grid as
        color image, as well as a list of the individual images generated (row-wise).
    '''
    figure = np.zeros((height * (n if bound_y > 0 else 1), width * (n if bound_x > 0 else 1), 3))
    grid_x = np.linspace(-bound_x, bound_x, n) if bound_x > 0 else [0]
    grid_y = np.linspace(-bound_y, bound_y, n) if bound_y > 0 else [0]
    individual_outputs = []

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = base_vec.copy()
            z_sample[axis_x] = xi # SD is 1
            z_sample[axis_y] = yi # SD is 1
            
            x_decoded = decoder.predict(np.expand_dims(z_sample, axis=0))
            sample = np.clip(x_decoded[0], 0, 1)
            figure[i * height: (i + 1) * height, j * width: (j + 1) * width] = sample
            individual_outputs.append(sample)

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.xlabel(desc_x)
    plt.ylabel(desc_y)
    if file_out is not None:
        plt.savefig(file_out, dpi=200, bbox_inches='tight')
    return figure, individual_outputs

```




```python
dim1 = 'Male' 
dim2 = 'Smiling'

img = skimage.io.imread('data/img_align_celeba/125526.jpg')
z = encode_image(img.astype(np.float32)/255, None, encoder, HEIGHT, WIDTH, BATCH_SIZE)
base_vec = np.array(list(z[0]))

rendering, _ = display_manifold(
    decoder, 
    HEIGHT, 
    WIDTH, 
    base_vec, 
    bound_x=2, 
    bound_y=2, 
    axis_x=1, 
    axis_y=14, 
    desc_x = dim2, 
    desc_y = dim1,
    n=10,
    file_out = 'rendering_celeba_' + dim1.lower() + '_' + dim2.lower() + '.png')
```



![png](cs109b_hw7_web_files/cs109b_hw7_web_36_0.png)


**1.B.5.** Generate and visualise around 15 celebrity faces not in your training set.  How do the generated faces compare in quality to celebrity faces from the training samples?



```python
np.random.seed(1)
NUM_SAMPLES = 16
plt.figure(figsize=(20, 5))

for i in range(NUM_SAMPLES):
    noise = np.random.randn(1, LATENT_DIM) 
    pred_raw = decode_embedding(noise, None, decoder)[0]
    pred = (pred_raw * 0.5 + 0.5)
    plt.subplot(2, 8, i + 1)
    plt.imshow(pred_raw)
    
plt.show()
```


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



![png](cs109b_hw7_web_files/cs109b_hw7_web_38_1.png)


Even though the pictures above can capture the key features of faces, likes eyes, nose, sex, color, and hair style etc, the quality overall is stilll way worse than the quality of the training samples. Some of the faces are blured and they all lack the background details. I think the model can be improved by enhancing the architecture of the VAE, tuning better hyperparameters and training with more epoches. 

### DCGAN Model 

**1.C.1.** Create and compile a DCGAN model for the celebrity faces dataset.  Print summaries for the discriminator and generator models.



```python
SPATIAL_DIM = 64 # Spatial dimensions of the images.
LATENT_DIM = 100 # Dimensionality of the noise vector.
BATCH_SIZE = 32 # Batchsize to use for training.
DISC_UPDATES = 1  # Number of discriminator updates per training iteration.
GEN_UPDATES = 1 # Nmber of generator updates per training iteration.

FILTER_SIZE = 5 # Filter size to be applied throughout all convolutional layers.
NUM_LOAD = 10000 # Number of images to load from CelebA. Fit also according to the available memory on your machine.
NET_CAPACITY = 16 # General factor to globally change the number of convolutional filters.

PROGRESS_INTERVAL = 80 # Number of iterations after which current samples will be plotted.
ROOT_DIR = 'visualization' # Directory where generated samples should be saved to.

if not os.path.isdir(ROOT_DIR):
    os.mkdir(ROOT_DIR)
```




```python
def add_encoder_block(x, filters, filter_size):
    x = Conv2D(filters, filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)
    return x
```




```python
def build_discriminator(start_filters, spatial_dim, filter_size):
    inp = Input(shape=(spatial_dim, spatial_dim, 3))
    
    # Encoding blocks downsample the image.
    x = add_encoder_block(inp, start_filters, filter_size)
    x = add_encoder_block(x, start_filters * 2, filter_size)
    x = add_encoder_block(x, start_filters * 4, filter_size)
    x = add_encoder_block(x, start_filters * 8, filter_size)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inp, outputs=x)
```




```python
def add_decoder_block(x, filters, filter_size):
    x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.3)(x)
    return x
```




```python
def build_generator(start_filters, filter_size, latent_dim):
    inp = Input(shape=(latent_dim,))
    
    # Projection.
    x = Dense(4 * 4 * (start_filters * 8), input_dim=latent_dim)(inp)
    x = BatchNormalization()(x)
    x = Reshape(target_shape=(4, 4, start_filters * 8))(x)
    
    # Decoding blocks upsample the image.
    x = add_decoder_block(x, start_filters * 4, filter_size)
    x = add_decoder_block(x, start_filters * 2, filter_size)
    x = add_decoder_block(x, start_filters, filter_size)
    x = add_decoder_block(x, start_filters, filter_size)    
    
    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
    return keras.Model(inputs=inp, outputs=x)
```




```python
def construct_models(verbose=False):
    # 1. Build discriminator.
    discriminator = build_discriminator(NET_CAPACITY, SPATIAL_DIM, FILTER_SIZE)
    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['mae'])

    # 2. Build generator.
    generator = build_generator(NET_CAPACITY, FILTER_SIZE, LATENT_DIM)

    # 3. Build full GAN setup by stacking generator and discriminator.
    gan = keras.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False # Fix the discriminator part in the full setup.
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['mae'])

    if verbose: # Print model summaries for debugging purposes.
        generator.summary()
        discriminator.summary()
        gan.summary()
    return generator, discriminator, gan
```




```python
def plot_image(x):
    plt.imshow(x * 0.5 + 0.5)
```




```python
def run_training(start_it=0, num_epochs=1000):
    config_name = 'gan_cap' + str(NET_CAPACITY) + '_batch' + str(BATCH_SIZE) + '_filt' + str(FILTER_SIZE) + '_disc' + str(DISC_UPDATES) + '_gen' + str(GEN_UPDATES)
    folder = os.path.join(ROOT_DIR, config_name)

    if not os.path.isdir(folder):
        os.mkdir(folder)
    avg_loss_discriminator = []
    avg_loss_generator = []
    total_it = start_it

    for epoch in range(num_epochs):
        loss_discriminator = []
        loss_generator = []
        for it in range(200): 

            # Update discriminator.
            for i in range(DISC_UPDATES): 
                # Fetch real examples (you could sample unique entries, too).
                
                imgs_real = np.empty(shape=(BATCH_SIZE, SPATIAL_DIM, SPATIAL_DIM, 3))
                for i in range(BATCH_SIZE):
                    file = np.random.choice(files)
                    img = skimage.io.imread(file)
                    if img.shape[0] != SPATIAL_DIM or img.shape[1] != SPATIAL_DIM:
                        img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
                    img = img.astype(np.float32) / 127.5 - 1.0
                    imgs_real[i] = img
                    #plot_image(img)

                # Generate fake examples.
                noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
                imgs_fake = generator.predict(noise)

                d_loss_real = discriminator.train_on_batch(imgs_real, np.ones([BATCH_SIZE]))[1]
                d_loss_fake = discriminator.train_on_batch(imgs_fake, np.zeros([BATCH_SIZE]))[1]
            
            # Progress visualizations.
            if total_it % PROGRESS_INTERVAL == 0:
                plt.figure(figsize=(5,2))
                # We sample separate images.
                num_vis = min(BATCH_SIZE, 8)
                imgs_real = np.empty(shape=(num_vis, SPATIAL_DIM, SPATIAL_DIM, 3))
                for i in range(num_vis):
                    file = np.random.choice(files)
                    img = skimage.io.imread(file)
                    if img.shape[0] != SPATIAL_DIM or img.shape[1] != SPATIAL_DIM:
                        img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
                    img = img.astype(np.float32) / 127.5 - 1.0
                    imgs_real[i] = img
                    #plot_image(img)
                
                
                noise = np.random.randn(num_vis, LATENT_DIM)
                imgs_fake = generator.predict(noise)
                for obj_plot in [imgs_fake, imgs_real]:
                    plt.figure(figsize=(num_vis * 3, 3))
                    for b in range(num_vis):
                        disc_score = float(discriminator.predict(np.expand_dims(obj_plot[b], axis=0))[0])
                        plt.subplot(1, num_vis, b + 1)
                        plt.title(str(round(disc_score, 3)))
                        plot_image(obj_plot[b]) 
                    if obj_plot is imgs_fake:
                        plt.savefig(os.path.join(folder, str(total_it).zfill(10) + '.jpg'), format='jpg', bbox_inches='tight')
                    plt.show()  

            # Update generator.
            loss = 0
            y = np.ones([BATCH_SIZE, 1]) 
            for j in range(GEN_UPDATES):
                noise = np.random.randn(BATCH_SIZE, LATENT_DIM)
                loss += gan.train_on_batch(noise, y)[1]

            loss_discriminator.append((d_loss_real + d_loss_fake) / 2.)        
            loss_generator.append(loss / GEN_UPDATES)
            total_it += 1

        # Progress visualization.
        clear_output(True)
        print('Epoch', epoch)
        avg_loss_discriminator.append(np.mean(loss_discriminator))
        avg_loss_generator.append(np.mean(loss_generator))
        plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator)
        plt.plot(range(len(avg_loss_generator)), avg_loss_generator)
        plt.legend(['discriminator loss', 'generator loss'])
        plt.show()
```




```python
generator, discriminator, gan = construct_models(verbose=True)
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         (None, 100)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 2048)              206848    
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 2048)              8192      
    _________________________________________________________________
    reshape_2 (Reshape)          (None, 4, 4, 128)         0         
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 8, 8, 64)          204864    
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 8, 8, 64)          256       
    _________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)    (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_transpose_2 (Conv2DTr (None, 16, 16, 32)        51232     
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 16, 16, 32)        128       
    _________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)    (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_transpose_3 (Conv2DTr (None, 32, 32, 16)        12816     
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 32, 32, 16)        64        
    _________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)    (None, 32, 32, 16)        0         
    _________________________________________________________________
    conv2d_transpose_4 (Conv2DTr (None, 64, 64, 16)        6416      
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 64, 64, 16)        64        
    _________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)    (None, 64, 64, 16)        0         
    _________________________________________________________________
    conv2d_26 (Conv2D)           (None, 64, 64, 3)         1203      
    =================================================================
    Total params: 492,083
    Trainable params: 487,731
    Non-trainable params: 4,352
    _________________________________________________________________
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         (None, 64, 64, 3)         0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 64, 64, 16)        1216      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 64, 64, 16)        64        
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, 32, 32, 16)        6416      
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 32, 32, 16)        64        
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 16)        0         
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 32, 32, 32)        12832     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 32, 32, 32)        128       
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 16, 16, 32)        25632     
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 16, 16, 32)        128       
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 16, 16, 32)        0         
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 16, 16, 64)        51264     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 16, 16, 64)        256       
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 8, 8, 64)          102464    
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 8, 8, 64)          256       
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_24 (Conv2D)           (None, 8, 8, 128)         204928    
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 8, 8, 128)         512       
    _________________________________________________________________
    conv2d_25 (Conv2D)           (None, 4, 4, 128)         409728    
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 4, 4, 128)         512       
    _________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)    (None, 4, 4, 128)         0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 128)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 1,632,098
    Trainable params: 815,569
    Non-trainable params: 816,529
    _________________________________________________________________
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    model_5 (Model)              (None, 64, 64, 3)         492083    
    _________________________________________________________________
    model_4 (Model)              (None, 1)                 816529    
    =================================================================
    Total params: 1,308,612
    Trainable params: 487,731
    Non-trainable params: 820,881
    _________________________________________________________________


**1.C.2.** Train your model on the images in the celeba dataset.



```python
```


    Epoch 3



![png](cs109b_hw7_web_files/cs109b_hw7_web_51_1.png)



    <Figure size 360x144 with 0 Axes>



![png](cs109b_hw7_web_files/cs109b_hw7_web_51_3.png)



![png](cs109b_hw7_web_files/cs109b_hw7_web_51_4.png)



    <Figure size 360x144 with 0 Axes>



![png](cs109b_hw7_web_files/cs109b_hw7_web_51_6.png)



![png](cs109b_hw7_web_files/cs109b_hw7_web_51_7.png)




```python
gan.load_weights('models/celeba_gan/gan225.w')
```


**1.C.3.** Generate and visualise around 15 celebrity faces.  How do the generated faces compare in quality to celebrity faces from the training samples? How do they compare in quality to the faces generated via VAE?



```python
np.random.seed(0)
NUM_SAMPLES = 16
plt.figure(figsize=(20, 5))

for i in range(NUM_SAMPLES):
    noise = np.random.randn(1, LATENT_DIM) 
    pred_raw = generator.predict(noise)[0]
    pred = (pred_raw * 0.5 + 0.5)
    plt.subplot(2, 8, i + 1)
    plt.imshow(pred)
plt.show()
```



![png](cs109b_hw7_web_files/cs109b_hw7_web_54_0.png)


In term of quality ranking, training sample > the pictures generated from VAE > the pictures generated from GAN. The pictures from VAE are more real, have richer details, and are more like real pictures vs the pictures from GAN which are more like paints of art. With that being said, the images from GAN do have the key features of face, including eyes, hair, mouth, and nose etc. Given that the pictures are purely generated from noise, they are both quite impressive in my opinion.
