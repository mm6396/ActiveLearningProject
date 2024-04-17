# Jeremy Lim
# jlim@wpi.edu
# Quick testbench framework, to help guide us in implementation.

# Basically, once this testbench is up and running, we can tweak/re-tweak DRLA using the training set to optimize performance.
# NOTE: NOT a good idea to tweak the classifier, only the AL method!!!

import sys, os

import numpy as np
from matplotlib import pyplot as plt

from ActiveLearningMethods import *


def train_on_AL_approach(image_classifier: object, al_instance: ActiveLearningMethod, Dataset_params: object, other_params: object):
    """
    Given an Active Learning Method and a Dataset, train our image_classifier according to the ActiveLearning strategy.
    """
    # prepare starting labels (Follow DRLA paper for comparison on 2 of our datasets, we'll decide ourselves on cancer MNIST.

    # Start epoch loop:

    # Classifier predicts on all data (labeled or unlabeled) (State)

    # call al_instance.choose_n_samples() to decide which samples to annotate.

    # Update classifier for 1 epoch on all of the labeled samples.

    # Classifier predicts again on all data (labeled or unlabeled) (New_State)

    # Call al_instance.update_on_new_state(). This will either do nothing for most methods, or update Actor/Critic for DRLA.

    # End epoch loop.........

    # Could also be a tuple if we want.
    someDataStructureContainingManyMetrics = {"Stuff": 123}

    return someDataStructureContainingManyMetrics

def main():

    # This script should be hardcoded into Train mode most of the time, but can be configured to do the Test dataset at the end.

    # Choose/prepare dataset (Chose 1 of the 3. Or I guess we could go through all 3)
    # ASSUME REPOSITORY DIRECTORY STRUCTURE (and data is available!)
    # We can port this to Turing or some other compute service if needed then by just cloning git repository!

    # Instantiate Resnet50 model

    # Instantiate class for each of the Active Learning approaches we'll be comparing

    # Loop: For each AL approach, call train_on_AL_approach(parameters). Collect metrics from this run.

    # Build table - follow intervals like in DRLA paper

    # Build per-epoch chart - do accuracy, sensitivity, specificity

    # Save charts, data for this set of runs.

    pass


if __name__ == "__main__":
    main()