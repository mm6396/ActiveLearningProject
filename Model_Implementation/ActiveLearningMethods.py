# Jeremy Lim
# jlim@wpi.edu
# Currently adding abstract class that all of our active learning methods must implement.
import abc

import numpy as np

class ActiveLearningMethod(abc.ABC):
    # All active learning methods must subclass this!

    def __init__(self):
        """
        Normal constructor for class. Subclass and pass in needed samples for the specific approach.
        """
        raise NotImplemented

    def choose_n_samples(self, sample_num, state, labeled_mask):
        """
        Based on the input state, choose samples to annotate.
        Let k = the number of classes. Let n = the total number of samples.
        :param sample_num: The number of samples to select for annotation. The annotator should select up to this many;
        it will select less if the unlabeled pool is smaller than this.
        :param state: The classifier prediction on all samples (labeled or unlabeled) in the pool.
        So a numpy array of shape nxk.
        :param labeled_mask: A mask that denotes which samples (1) have been "annotated" and which samples (0)
        have not been revealed to the classifier. Numpy array of size n.
        :return: Numpy array of indices (indexing into state), denoting which samples to reveal labels for.
        So a numpy array of size sample_num
        """
        raise NotImplemented

    def update_on_new_state(self, new_state, new_state_labeled_mask, previous_state, previous_state_labeled_mask):
        """
        For most methods, this does nothing. For DRLA, update and train actor/critic based on the states.
        Let k = the number of classes. Let n = the total number of samples.
        :param new_state: Classifier prediction on all samples, based on updated classifier. So a numpy array of shape nxk.
        :param new_state_labeled_mask: A mask that denotes which samples (1) have been "annotated" and which samples (0)
        have not been revealed to the classifier. Numpy array of size n.
        :param previous_state: Classifier prediction on all samples, based on the classifier before the weights were updated.
        So a numpy array of shape nxk.
        :param previous_state_labeled_mask: A mask that denotes which samples (1) have been "annotated" and which samples (0)
        have not been revealed to the classifier. Numpy array of size n.
        :return: None. This only updates internal state.
        """
        pass

    # zaher
    # random

    # Jeremy
    # entropy

    # Adish
    # Least confidence

    # Still figuring out
    # DRLA

    # margin sampling

    # FUSION - low priority

# class DRLA(ActiveLearningMethod):
#     self.actor_Model = "abs"