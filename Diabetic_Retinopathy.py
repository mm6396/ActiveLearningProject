from tensorflow import lite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random, os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split

#
df = pd.read_csv(r'../rawdata/train.csv') #assumes data is put into a file called rawdata
# df = pd.read_csv(r'../input/diabetic-retinopathy-224x224-gaussian-filtered/train.csv')

diagnosis_dict_binary = {
    0: 'No_DR',
    1: 'DR',
    2: 'DR',
    3: 'DR',
    4: 'DR'
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}


df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)
df.head()

#
df['type'].value_counts().plot(kind='barh')

#
from sklearn.model_selection import train_test_split

half1, half2 = train_test_split(df, test_size=0.5, random_state=42)  # random_state

# Splitting the first half into training (80%) and testing (20%) sets
train1, test1 = train_test_split(half1, test_size=0.2, stratify=half1['type'], random_state=42)

# Splitting the second half into training (80%) and testing (20%) sets
train2, test2 = train_test_split(half2, test_size=0.2, stratify=half2['type'], random_state=42)

# Printing the counts of each type in each dataset
print("For First Training Dataset :")
print(train1['type'].value_counts(), '\n')
print("For First Testing Dataset :")
print(test1['type'].value_counts(), '\n')
print("For Second Training Dataset :")
print(train2['type'].value_counts(), '\n')
print("For Second Testing Dataset :")
print(test2['type'].value_counts(), '\n')

#
import shutil
base_dir = '..\Skin_Cancer_MNIST'

train1_dir = os.path.join(base_dir, 'train1')
# print(train1_dir)
train2_dir = os.path.join(base_dir, 'train2')

test1_dir = os.path.join(base_dir, 'test1')
test2_dir = os.path.join(base_dir, 'test2')

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

if os.path.exists(train1_dir):
    shutil.rmtree(train1_dir)
os.makedirs(train1_dir)

if os.path.exists(train2_dir):
    shutil.rmtree(train2_dir)
os.makedirs(train2_dir)

if os.path.exists(test1_dir):
    shutil.rmtree(test1_dir)
os.makedirs(test1_dir)

if os.path.exists(test2_dir):
    shutil.rmtree(test2_dir)
os.makedirs(test2_dir)


#src_dir = r'../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images'
src_dir = r'../rawdata/gaussian_filtered_images/gaussian_filtered_images'
for index, row in train1.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(train1_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
    
    
for index, row in train2.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(train2_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)


for index, row in test1.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(test1_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)
    
for index, row in test2.iterrows():
    diagnosis = row['type']
    binary_diagnosis = row['binary_type']
    id_code = row['id_code'] + ".png"
    srcfile = os.path.join(src_dir, diagnosis, id_code)
    dstfile = os.path.join(test2_dir, binary_diagnosis)
    os.makedirs(dstfile, exist_ok = True)
    shutil.copy(srcfile, dstfile)

    #TO make sure images show up
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train1_path = '../Skin_Cancer_MNIST/train1'
# train2_path = '../Skin_Cancer_MNIST/train2'
# test1_path = '../Skin_Cancer_MNIST/test1'
# test2_path = '../Skin_Cancer_MNIST/test2'

# train1_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train1_path, target_size=(224,224), shuffle = True)
# train2_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train2_path, target_size=(224,224), shuffle = True)
# test1_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test1_path, target_size=(224,224), shuffle = False)
# test1_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test1_path, target_size=(224,224), shuffle = False)

