
from tensorflow import lite
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random, os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import shutil

# Load the data
df = pd.read_csv('../rawdata/train.csv')

# Mapping diagnosis to categorical types
diagnosis_dict_binary = {
    # 0: 'No_DR',
    # 1: 'DR',
    # 2: 'DR',
    # 3: 'DR',
    # 4: 'DR'
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

diagnosis_dict = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR',
}

df['binary_type'] = df['diagnosis'].map(diagnosis_dict_binary.get)
df['type'] = df['diagnosis'].map(diagnosis_dict.get)

# Splitting data
half1, half2 = train_test_split(df, test_size=0.5, random_state=42)
train1, test1 = train_test_split(half1, test_size=0.2, stratify=half1['type'], random_state=42)
train2, test2 = train_test_split(half2, test_size=0.2, stratify=half2['type'], random_state=42)

# Creating directories
base_dir = 'Diabetic_Retinopathy'
directories = ['train1', 'train2', 'test1', 'test2']
for directory in directories:
    dir_path = os.path.join(base_dir, directory)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

src_dir = '../rawdata/gaussian_filtered_images/gaussian_filtered_images'

# Function to load images into numpy arrays
def load_images(dataframe, src_dir, target_dir):
    images = []
    labels = []
    for index, row in dataframe.iterrows():
        diagnosis = row['type']
        binary_diagnosis = row['binary_type']
        id_code = row['id_code'] + ".png"
        srcfile = os.path.join(src_dir, diagnosis, id_code)
        dstfile = os.path.join(target_dir, binary_diagnosis)
        os.makedirs(dstfile, exist_ok=True)
        shutil.copy(srcfile, dstfile)
        
        # Read and append the image data
        img = imread(srcfile)
        images.append(img)
        labels.append(binary_diagnosis)

    return np.array(images), np.array(labels)

# Loading images for each dataset
train1_images, train1_labels = load_images(train1, src_dir, os.path.join(base_dir, 'train1'))
train2_images, train2_labels = load_images(train2, src_dir, os.path.join(base_dir, 'train2'))
test1_images, test1_labels = load_images(test1, src_dir, os.path.join(base_dir, 'test1'))
test2_images, test2_labels = load_images(test2, src_dir, os.path.join(base_dir, 'test2'))
