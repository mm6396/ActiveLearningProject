# Jeremy Lim
# jlim@wpi.edu
# Script for building our PatchCamelyon dataset.
# Source: https://github.com/basveeling/pcam?tab=readme-ov-file

import sys, os
from collections import OrderedDict  # ordered so that we can reproduce consistent behavior.
import random

import h5py
import pandas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

VAL_FRACTION = 0.2  # This portion will be
UNBALANCE_LIMIT = 0.03  # Should be within +/- of this fraction of 0.5 class balance
RAND_SEED = 59864  # Random seed. Use to build the exact same datasets on different machines!

# For comparing to the original paper.
COMPARE_TRAIN_SIZE = 250
COMPARE_VAL_SIZE = 63
SUBSAMPLE_MARGIN_MULTIPLIER = 1.1  # we give ourselves some margin so we can easily split and subsample down to 250/63

IMG_SIZE = 96
IMG_CHANNELS = 3

# source webpage:


META_TRAIN = 'raw/camelyonpatch_level_2_split_train_meta.csv'
META_VAL = 'raw/camelyonpatch_level_2_split_valid_meta.csv'
META_TEST = 'raw/camelyonpatch_level_2_split_test_meta.csv'

# file paths
TRAIN_X_H5_PATH = "raw/camelyonpatch_level_2_split_train_x.h5"
TRAIN_Y_H5_PATH = "raw/camelyonpatch_level_2_split_train_y.h5"

VAL_X_H5_PATH = "raw/camelyonpatch_level_2_split_valid_x.h5"
VAL_Y_H5_PATH = "raw/camelyonpatch_level_2_split_valid_y.h5"

TEST_X_H5_PATH = "raw/camelyonpatch_level_2_split_test_x.h5"
TEST_Y_H5_PATH = "raw/camelyonpatch_level_2_split_test_y.h5"

# Code notes and inspiration to help me put this together:
# I'm rusty with pandas! : https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe

# I could use this, but it requires group distribution to be balanced sadly: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
# Will use this: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
# https://docs.h5py.org/en/stable/


def build_test_train_splits(val_fraction=VAL_FRACTION):

    # parse metadata
    train_diction = build_meta_dict(META_TRAIN)
    print("Items in training set: " + str(len(train_diction.keys())))
    val_diction = build_meta_dict(META_VAL)
    print("Items in validation set: " + str(len(val_diction.keys())))
    test_diction = build_meta_dict(META_TEST)
    print("Items in training set: " + str(len(test_diction.keys())))

    # data_diction.update(val_diction)
    # data_diction.update(test_diction)
    # print("Items in combined set: " + str(len(data_diction.keys())))

    # Format for GroupShuffleSplitters
    x_arr = []
    x_set_arr = []  # second dimension; keep track of whether it came from the original train, val or test
    # train: 0
    # val: 1
    # test: 2
    y_arr = []
    group_num = 0
    group_arr = []

    # Training
    print("Flatten training...")
    for k in range(len(train_diction.keys())):
        record = train_diction.popitem()[1]
        x_arr = x_arr + record['positive_list']
        for positive_id in record['positive_list']:
            y_arr.append(1)
            group_arr.append(group_num)
            x_set_arr.append(0)

        x_arr = x_arr + record['negative_list']
        for negative_id in record['negative_list']:
            y_arr.append(0)
            group_arr.append(group_num)
            x_set_arr.append(0)

        group_num += 1

    # Validation
    print("Flatten validation...")
    for k in range(len(val_diction.keys())):
        record = val_diction.popitem()[1]
        x_arr = x_arr + record['positive_list']
        for positive_id in record['positive_list']:
            y_arr.append(1)
            group_arr.append(group_num)
            x_set_arr.append(1)

        x_arr = x_arr + record['negative_list']
        for negative_id in record['negative_list']:
            y_arr.append(0)
            group_arr.append(group_num)
            x_set_arr.append(1)

        group_num += 1

    # Test
    print("Flatten test...")
    for k in range(len(test_diction.keys())):
        record = test_diction.popitem()[1]
        x_arr = x_arr + record['positive_list']
        for positive_id in record['positive_list']:
            y_arr.append(1)
            group_arr.append(group_num)
            x_set_arr.append(2)

        x_arr = x_arr + record['negative_list']
        for negative_id in record['negative_list']:
            y_arr.append(0)
            group_arr.append(group_num)
            x_set_arr.append(2)

        group_num += 1

    x_arr = np.array(x_arr).reshape((-1, 1))
    x_set_arr = np.array(x_set_arr).reshape((-1, 1))

    x_arr = np.concatenate((x_arr, x_set_arr), axis=1)
    y_arr = np.array(y_arr)
    group_arr = np.array(group_arr)

    # starting splitting routine

    finished_splitting = False

    train_test_splitter = GroupShuffleSplit(n_splits=1000, test_size=0.5, train_size=0.5,
                                       random_state=RAND_SEED)
    val_sub_splitter = GroupShuffleSplit(n_splits=1000, test_size=val_fraction, train_size=1.0-val_fraction,
                                       random_state=RAND_SEED)
    # Random splits to try.
    train_test_iterator = train_test_splitter.split(x_arr, y_arr, group_arr)

    while not finished_splitting:
        print("Trying Train/Test split: ")
        train_idxs, test_idxs = split_enforce_balance(y_arr, train_test_iterator, 0.5, unbalance_limit=UNBALANCE_LIMIT)
        try:
            train_x = x_arr[train_idxs]
            train_y = y_arr[train_idxs]
            train_group = group_arr[train_idxs]

            test_x = x_arr[test_idxs]
            test_y = y_arr[test_idxs]
            test_group = group_arr[test_idxs]

            # Train/Test Splits computed. Now create sub-splits for each validation set.
            print("Sub-split for train set (20% validation): ")
            train_train_idxs, train_val_idxs = split_enforce_balance(train_y, val_sub_splitter.split(train_x, train_y, train_group),
                                                                     val_fraction, unbalance_limit=UNBALANCE_LIMIT)

            train_train_x = train_x[train_train_idxs]
            train_train_y = train_y[train_train_idxs]
            train_train_group = train_group[train_train_idxs]

            train_val_x = train_x[train_val_idxs]
            train_val_y = train_y[train_val_idxs]
            train_val_group = train_group[train_val_idxs]

            print("Sub-split for test set (20% validation): ")
            test_train_idxs, test_val_idxs = split_enforce_balance(test_y, val_sub_splitter.split(test_x, test_y, test_group),
                                                                   val_fraction, unbalance_limit=UNBALANCE_LIMIT)

            test_train_x = train_x[test_train_idxs]
            test_train_y = train_y[test_train_idxs]
            test_train_group = test_group[test_train_idxs]

            test_val_x = train_x[test_val_idxs]
            test_val_y = train_y[test_val_idxs]
            test_val_group = test_group[test_val_idxs]


        except StopIteration as e:
            # This means we couldn't balance a split; so we need to redo much of the whole process.
            print("Could not build a split that works. Retrying...")
            continue

        finished_splitting = True


    print("Building 250/250 groups (with validation sets)...")
    # Build splits to emulate the 250/250 split case.
    # so technically, we'll split to 313/313 to build a 20% validation split that has 250 samples as training data.
    comparison_sub_sampler = StratifiedShuffleSplit(n_splits=1000, test_size=int(COMPARE_TRAIN_SIZE/(1.0-val_fraction)*SUBSAMPLE_MARGIN_MULTIPLIER), random_state=RAND_SEED)
    comparison_train_iter = comparison_sub_sampler.split(train_x, train_y, train_group)

    unused, comp_train_idxs = next(comparison_train_iter)
    while not checkGroupBalance(train_y, comp_train_idxs, UNBALANCE_LIMIT):
        print("Class balance not close enough. Retrying...")
        unused, comp_train_idxs = next(comparison_train_iter)

    comparison_train_x = train_x[comp_train_idxs]
    comparison_train_y = train_y[comp_train_idxs]
    comparison_train_group = train_group[comp_train_idxs]

    comparison_test_iter = comparison_sub_sampler.split(test_x, test_y, test_group)

    unused, comp_test_idxs = next(comparison_test_iter)
    while not checkGroupBalance(test_y, comp_test_idxs, UNBALANCE_LIMIT):
        print("Class balance not close enough. Retrying...")
        unused, comp_test_idxs = next(comparison_test_iter)

    comparison_test_x = test_x[comp_test_idxs]
    comparison_test_y = test_y[comp_test_idxs]
    comparison_test_group = test_group[comp_test_idxs]

    comp_train_val_grouper = GroupShuffleSplit(n_splits=1000, train_size=0.8, random_state=RAND_SEED)
    comp_train_val_iter = comp_train_val_grouper.split(comparison_train_x, comparison_train_y, comparison_train_group)

    comp_train_train_idx, comp_train_val_idx = next(comp_train_val_iter)
    while not (checkGroupBalance(comparison_train_y, comp_train_train_idx, UNBALANCE_LIMIT) and checkGroupBalance(comparison_train_y, comp_train_val_idx, UNBALANCE_LIMIT)
               and len(comp_train_train_idx) >= COMPARE_TRAIN_SIZE):
        print("Class balance not close enough. Retrying...")
        comp_train_train_idx, comp_train_val_idx = next(comp_train_val_iter)

    comp_train_train_x = comparison_train_x[comp_train_train_idx]
    comp_train_train_y = comparison_train_y[comp_train_train_idx]

    comp_train_train_x, comp_train_train_y = balanced_stratify_to_size(comp_train_train_x, comp_train_train_y,
                                                                     newsize=COMPARE_TRAIN_SIZE)

    comp_train_val_x = comparison_train_x[comp_train_val_idx]
    comp_train_val_y = comparison_train_y[comp_train_val_idx]

    comp_train_val_x, comp_train_val_y = balanced_stratify_to_size(comp_train_val_x, comp_train_val_y,
                                                                     newsize=COMPARE_VAL_SIZE)

    comp_test_val_iter = comp_train_val_grouper.split(comparison_test_x, comparison_test_y, comparison_test_group)

    comp_test_train_idx, comp_test_val_idx = next(comp_test_val_iter)
    while not (checkGroupBalance(comparison_test_y, comp_test_train_idx, UNBALANCE_LIMIT) and checkGroupBalance(comparison_test_y, comp_test_val_idx, UNBALANCE_LIMIT)
               and len(comp_test_train_idx) >= COMPARE_TRAIN_SIZE):
        print("Class balance not close enough. Retrying...")
        comp_test_train_idx, comp_test_val_idx = next(comp_test_val_iter)

    comp_test_train_x = comparison_test_x[comp_test_train_idx]
    comp_test_train_y = comparison_test_y[comp_test_train_idx]

    comp_test_train_x, comp_test_train_y = balanced_stratify_to_size(comp_test_train_x, comp_test_train_y,
                                                                     newsize=COMPARE_TRAIN_SIZE)

    comp_test_val_x = comparison_test_x[comp_test_val_idx]
    comp_test_val_y = comparison_test_y[comp_test_val_idx]

    comp_test_val_x, comp_test_val_y = balanced_stratify_to_size(comp_test_val_x, comp_test_val_y,
                                                                     newsize=COMPARE_VAL_SIZE)

    print("Packing datasets into h5 files.")
    # build the relevant hdf5 files.



    # Training - full
    build_hdf5_file("output/train_train", train_train_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)
    build_hdf5_file("output/train_val", train_val_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)

    # Testing - full
    build_hdf5_file("output/test_train", test_train_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)
    build_hdf5_file("output/test_val", test_val_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)

    # Training - 250
    build_hdf5_file("output/train250_train", comp_train_train_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)
    build_hdf5_file("output/train250_val", comp_train_val_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)

    # Testing - 250
    build_hdf5_file("output/test250_train", comp_test_train_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)
    build_hdf5_file("output/test250_val", comp_test_val_x, TRAIN_X_H5_PATH, TRAIN_Y_H5_PATH, VAL_X_H5_PATH, VAL_Y_H5_PATH,
                    TEST_X_H5_PATH, TEST_Y_H5_PATH)


def build_meta_dict(fpath):
    df = pandas.read_csv(fpath)

    # map whole slide names to a data structure to track indices,
    whole_slide_dict = OrderedDict()
    # each entry:
    # {
    #     'count': int, # Number of samples from this slide
    #     'id_list': list # List of ids for this slide.
    #     'positive_list': list # positive ids
    #     'negative_list': list # negative ids
    # }

    # not sure if needed?
    # df = df.reset_index()
    total_negatives = 0
    total_positives = 0

    df.reset_index()

    for index, row in df.iterrows():
        slide_name = row['wsi']

        if slide_name in whole_slide_dict:
            whole_slide_dict[slide_name]['count'] += 1
        else:
            whole_slide_dict[slide_name] = {
                'count': 1,
                'positive_list': [],
                'negative_list': [],
            }

        # Following dataset convention, only do "tumor" if the center contains a tumor, not if edges of the image do.
        if row['center_tumor_patch']:
            whole_slide_dict[slide_name]['positive_list'].append(index)
            total_positives += 1
        else:
            whole_slide_dict[slide_name]['negative_list'].append(index)
            total_negatives += 1

    print("Total Negatives: " + str(total_negatives))
    print("Total Positives: " + str(total_positives))

    return whole_slide_dict

def split_enforce_balance(y_arr, split_iterator, test_fraction, unbalance_limit=UNBALANCE_LIMIT):
    # Split a dataset into 2 parts by group, but re-run the stochastic procedure until we're relatively balanced.
    # Returns a (train, test) tuple of indices.

    # Iterate through a series of splits to test.
    # split_iterator = group_splitter.split(x_arr, y=y_arr, groups=group_arr)

    balanced = False
    split_runs = 0

    # TODO/NOTE: Ensure the splitter has a large number of split iterations specified
    while not balanced:
        split_runs += 1
        try:
            trainidxs, testidxs = next(split_iterator)
        except StopIteration as e:
            # If you can't split it this way, you need to go back up and retry the process!
            raise e

        # check for train/test approximate balance
        train_balance = len(trainidxs) / (len(trainidxs) + len(testidxs))
        print("Splitting attempt # " + str(split_runs) + "; train_balance: " + str(train_balance))
        if (train_balance > ((1.0-test_fraction) + unbalance_limit)) or (train_balance < ((1.0-test_fraction) - unbalance_limit)):
            continue

        balanced = checkGroupBalance(y_arr, trainidxs, unbalance_limit) and checkGroupBalance(y_arr, testidxs,
                                                                                              unbalance_limit)

    return trainidxs, testidxs


def checkGroupBalance(y_arr, indices, unbalance_limit=UNBALANCE_LIMIT):
    # Return true if it's balanced within the limit, false otherwise
    sampled_set = y_arr[indices]

    # Binary case makes this easy.
    positives = np.count_nonzero(sampled_set)
    negatives = len(sampled_set) - positives

    positive_balance = positives / len(sampled_set)

    print("Balance of positive class count: " + str(positive_balance))

    if positive_balance > (0.5 + unbalance_limit) or positive_balance < (0.5 - unbalance_limit):
        return False

    return True


def balanced_stratify_to_size(x, y, newsize, unbalance_limit=UNBALANCE_LIMIT, random_seed=RAND_SEED):
    # Subsample this dataset, keeping the class balance similar to what it was originally.
    # This does not worry about groups.

    stratify_splitter = StratifiedShuffleSplit(n_splits=1000, test_size=newsize, random_state=random_seed)
    split_iter = stratify_splitter.split(x, y)

    unused, newidx = next(split_iter)
    while not checkGroupBalance(y, newidx, unbalance_limit=unbalance_limit):
        unused, newidx = next(split_iter)

    return x[newidx], y[newidx]


def build_hdf5_file(fname, index_series, train_x_h5_path, train_y_h5_path, val_x_h5_path, val_y_h5_path,
                    test_x_h5_path, test_y_h5_path):

    print("Building files for: " + fname)

    train_x_file = h5py.File(train_x_h5_path,'r')
    train_y_file = h5py.File(train_y_h5_path,'r')
    val_x_file = h5py.File(val_x_h5_path,'r')
    val_y_file =h5py.File(val_y_h5_path,'r')
    test_x_file = h5py.File(test_x_h5_path,'r')
    test_y_file = h5py.File(test_y_h5_path, 'r')

    # create a new file to write to
    new_x_file = h5py.File(fname + "_x.h5", 'a')
    new_y_file = h5py.File(fname + "_y.h5", 'a')

    new_x_file.require_dataset('x', (len(index_series), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.uint8)
    new_y_file.require_dataset('y', (len(index_series), 1, 1, 1), dtype=np.uint8)


    # build the new dataset
    for idx in range(len(index_series)):
        if idx % 100 == 0:
            print(str(idx+1) + "/" + str(len(index_series)))

        if index_series[idx][1] == 0:  # train
            new_x_file['x'][idx, :, :, :] = train_x_file['x'][index_series[idx][0], :, :, :]
            new_y_file['y'][idx, :, :, :] = train_y_file['y'][index_series[idx][0], :, :, :]
        elif index_series[idx][1] == 1:  # val
            new_x_file['x'][idx, :, :, :] = val_x_file['x'][index_series[idx][0], :, :, :]
            new_y_file['y'][idx, :, :, :] = val_y_file['y'][index_series[idx][0], :, :, :]
        if index_series[idx][1] == 2:  # test
            new_x_file['x'][idx, :, :, :] = test_x_file['x'][index_series[idx][0], :, :, :]
            new_y_file['y'][idx, :, :, :] = test_y_file['y'][index_series[idx][0], :, :, :]

    # Remember to close all of the files!
    new_x_file.close()
    new_y_file.close()

    train_x_file.close()
    train_y_file.close()
    val_x_file.close()
    val_y_file.close()
    test_x_file.close()
    test_y_file.close()

def main():
    random.seed(RAND_SEED)

    if not os.path.exists("raw"):
        print("You need to create a 'raw' directory and put the extracted files from this webpage: https://github.com/basveeling/pcam?tab=readme-ov-file")

    os.makedirs("output", exist_ok=True)

    print("Building Train/Test split...")
    build_test_train_splits(val_fraction=VAL_FRACTION)

    print("Done")

if __name__ == "__main__":
    main()