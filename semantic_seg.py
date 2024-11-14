import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/carvana-image-masking-challenge/train/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!unzip /kaggle/input/carvana-image-masking-challenge/train.zip -d /kaggle/carvana-image-masking-challenge/

!unzip /kaggle/input/carvana-image-masking-challenge/train_masks.zip -d /kaggle/carvana-image-masking-challenge/

import glob
import pandas as pd
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Input, Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import utils
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
%matplotlib inline
from keras import backend as K

files_path = '../carvana-image-masking-challenge/train'
target_files_path = '../carvana-image-masking-challenge/train_masks'

data_files = {}
data_target = {}
data_files['files_path'] = []
data_target['target_files_path'] = []
data_files['files_path'] = list(glob.glob(files_path + "/*"))
data_target['target_files_path'] = list(glob.glob(target_files_path + "/*"))

data_files = pd.DataFrame(data_files)
data_target = pd.DataFrame(data_target)

def file_name(x):
    return x.split("/")[-1].split(".")[0]
data_files["file_name"] = data_files["files_path"].apply(lambda x: file_name(x))
data_target["file_name"] = data_target["target_files_path"].apply(lambda x: file_name(x)[:-5])

data = pd.merge(data_files, data_target, on = "file_name", how = "inner")

data.head()

n = int(round(data.shape[0] * 0.7,0))
data_train = data[0:n]
data_test = data[n:]

images_test = np.array([img_to_array(
                    load_img(img, target_size=(256,256))
                    ) for img in data_test['files_path'].values.tolist()])



images_train = np.array([img_to_array(
                    load_img(img, target_size=(256,256))
                    ) for img in data_train['files_path'].values.tolist()])

images_train = images_train.astype('float32')/255.0
images_test = images_test.astype('float32')/255.0

images_test_target = np.array([np.average(img_to_array(
                    load_img(img, target_size=(256,256))
                    )/255, axis=-1) for img in data_test['target_files_path'].values.tolist()])

images_train_target = np.array([np.average(img_to_array(
                    load_img(img, target_size=(256,256))
                    )/255, axis=-1) for img in data_train['target_files_path'].values.tolist()])

images_train_target = images_train_target[:,:,:,None]
images_test_target = images_test_target[:,:,:,None]

images_test_target[0].shape

import gc
gc.collect()

fig, axes = plt.subplots(ncols=2, figsize=(12, 12))
ax1, ax2 = axes
ax1.imshow(images_train[0]);
#ax1.set_grid(True);
ax1.set_xticks([]);
ax1.set_yticks([]);
ax1.set_title("Original Image Train")

ax2.imshow(np.squeeze(images_train_target[0]));
#ax2.set_grid(True);
ax2.set_xticks([]);
ax2.set_yticks([])
ax2.set_title("Mask")

