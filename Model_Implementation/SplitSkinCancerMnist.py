import pandas as pd
from pathlib import Path
from sklearn import model_selection
import collections
#from PIL import Image
#import matplotlib.pyplot as plt

import os

RAND_SEED = 2653


# get number of non-unique lesion_id's
def check_duplicates(df):
    unique_list = df.lesion_id.unique().tolist()
    num_duplicates = len(df) - len(unique_list)
    return num_duplicates


data_path = Path("../skin-cancer-mnist")

csv_path = data_path / 'HAM10000_metadata.csv'
scMnist_data = pd.read_csv(csv_path).set_index('image_id')
#print(scMnist_data.head())

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

scMnist_data.dx=scMnist_data.dx.astype('category',copy=True)
scMnist_data['label']=scMnist_data.dx.cat.codes # Convert the labels to numbers
scMnist_data['lesion_type'] = scMnist_data.dx.map(lesion_type_dict)
print(scMnist_data.head())

# {filename : path} for all files in both image folders
imageid_path_dict = {str(x).split(os.sep)[-1][:-4]: str(x) for x in list(data_path.glob('*/*.jpg'))}
#imageid_path_dict = {str(x).split('/')[-1]: str(x) for x in list(data_path.glob('*/*.jpg'))}
# use {filename: path} dict to select items from the correct folders
#scMnist_data['path'] = [Path(data_path/imageid_path_dict[fn].split('/')[3]/f'{fn}.jpg') for fn in scMnist_data.index.values]
scMnist_data['path'] = [Path(imageid_path_dict[fn]) for fn in scMnist_data.index.values]


# Determine duplicates
num_duplicates = check_duplicates(scMnist_data)
print(f'Duplicate lesion_ids: {num_duplicates} out of {len(scMnist_data)}')

test_set_fraction = 0.5

# Split the test and training sets
scMnist_train, scMnist_test = model_selection.train_test_split(scMnist_data, test_size=test_set_fraction, random_state=RAND_SEED)

# remove any duplicate images from the test set and check
scMnist_test = scMnist_test.drop_duplicates(subset='lesion_id', keep="first")
num_duplicates = check_duplicates(scMnist_test)
print(f'Duplicate lesion_ids in test set: {num_duplicates} out of {len(scMnist_test)}')

# remove any lesions from the train set that are also in the test set
scMnist_train = scMnist_train[~scMnist_train.lesion_id.isin(scMnist_test.lesion_id)]

# check test and train dfs have no shared `lesion_ids` or `image_ids`
check_lesion_ids = scMnist_test['lesion_id'].isin(scMnist_train['lesion_id']).value_counts()
check_image_ids = collections.Counter(scMnist_test.index.isin(scMnist_train.index))
#print(f'Test/train overlap? lesion_id: {int(check_lesion_ids) != len(scMnist_test)}, image_id: {check_image_ids[0] != len(scMnist_test)}')
print(f'Test/train overlap? lesion_id: {int(check_lesion_ids.iloc[0]) != len(scMnist_test)}, image_id: {check_image_ids[0] != len(scMnist_test)}')

# Check the balance of the train and test sets
for i, (df, title) in enumerate([(scMnist_train, 'Train'), (scMnist_test, 'Test')]):
    data = df['lesion_type'].value_counts()
    print(f'{title} distribution:')
    print(data)


# Split training data into validation and training sets
val_set_fraction = 0.2

scMnist_train, scMnist_val = model_selection.train_test_split(scMnist_train, test_size=val_set_fraction, random_state=RAND_SEED)

# remove any duplicate images from the val set and check
scMnist_val = scMnist_val.drop_duplicates(subset='lesion_id', keep="first")
num_duplicates = check_duplicates(scMnist_val)
print(f'Duplicate lesion_ids in val set: {num_duplicates} out of {len(scMnist_val)}')

# remove any lesions from the train set that are also in the val set
scMnist_train = scMnist_train[~scMnist_train.lesion_id.isin(scMnist_val.lesion_id)]

# check test and train dfs have no shared `lesion_ids` or `image_ids`
check_lesion_ids = scMnist_val['lesion_id'].isin(scMnist_train['lesion_id']).value_counts()
check_image_ids = collections.Counter(scMnist_val.index.isin(scMnist_train.index))
print(f'val/train overlap? lesion_id: {int(check_lesion_ids.iloc[0]) != len(scMnist_val)}, image_id: {check_image_ids[0] != len(scMnist_val)}')

# Check the balance of the train and test sets
for i, (df, title) in enumerate([(scMnist_train, 'Train'), (scMnist_val, 'Validation')]):
    data = df['lesion_type'].value_counts()
    print(f'{title} distribution:')
    print(data)


# Split test data into validation and training sets
scMnist_test, scMnist_testVal = model_selection.train_test_split(scMnist_test, test_size=val_set_fraction, random_state=RAND_SEED)

# remove any duplicate images from the val set and check
scMnist_testVal = scMnist_testVal.drop_duplicates(subset='lesion_id', keep="first")
num_duplicates = check_duplicates(scMnist_testVal)
print(f'Duplicate lesion_ids in val set: {num_duplicates} out of {len(scMnist_testVal)}')

# remove any lesions from the test set that are also in the testVal set
scMnist_test = scMnist_test[~scMnist_test.lesion_id.isin(scMnist_testVal.lesion_id)]

# check testVal and test data sets have no shared `lesion_ids` or `image_ids`
check_lesion_ids = scMnist_testVal['lesion_id'].isin(scMnist_test['lesion_id']).value_counts()
check_image_ids = collections.Counter(scMnist_testVal.index.isin(scMnist_test.index))
print(f'testVal/test overlap? lesion_id: {int(check_lesion_ids.iloc[0]) != len(scMnist_testVal)}, image_id: {check_image_ids[0] != len(scMnist_testVal)}')

# Check the balance of the test and testVal sets
for i, (df, title) in enumerate([(scMnist_test, 'Test'), (scMnist_testVal, 'Test Validation')]):
    data = df['lesion_type'].value_counts()
    print(f'{title} distribution:')
    print(data)



# print(image_path)
# image_id = 'ISIC_0032129'
# image_path = imageid_path_dict[image_id]

# # Open one image and display it
# image = Image.open(image_path)
# plt.imshow(image)
# plt.axis('off') 
# plt.show()


# print(scMnist_train.head())
# print(scMnist_val.head())
# print(scMnist_test.head())
# print(scMnist_testVal.head())