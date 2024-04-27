# Jeremy Lim
# jlim@wpi.edu
# All of our active learning strategies are implemented here.

import abc
import random  # for batch shuffling
import math

import numpy as np
import keras
from keras import layers
import tensorflow as tf

class ActiveLearningMethod(abc.ABC):
    # All active learning methods must subclass this!

    def __init__(self):
        """
        Normal constructor for class. Subclass and pass in needed samples for the specific approach.
        """
        pass

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
        :param previous_state_labeled_mask: A mask that denotes which samples (True) have been "annotated" and which samples (False)
        have not been revealed to the classifier. Numpy array of size n.
        :return: None. This only updates internal state.
        """
        pass



# Jeremy
# entropy
class EntropyStrategy(ActiveLearningMethod):

    # Simple; nothing needed for init!
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(in_arr):
        """
        Compute shannon entropy on a batch of softmax class "probabilities"
        :param in_arr: nxk matrix, n being the number of samples, k being the number of classes.
        :return: an array of length n, being the entropy computed for each sample.
        """
        # Entropy notes:
        # The minimum entropy is always zero, which corresponds to a 1-hot encoding with all
        # zeros except for the selected class being 1.
        # The maximum entropy is when the output is equally balanced between all classes.
        # So the max is

        # I'm using base 2 in this case.

        arr_log = np.log2(in_arr)
        # This may generate runtime warnings, but the following line will handle it and prevent -np.inf from propagating...
        arr_log[np.isinf(arr_log)] = 0  # handle -infinity.

        return -np.sum(np.log2(in_arr) * in_arr, axis=1)

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
        # Get entropy scores.
        # Using log2, but it doesn't matter because we rank them.

        action_arr = EntropyStrategy.entropy(state)

        # Remember the correct indices!
        sample_idxs = np.arange(0, state.shape[0], 1, dtype=np.int64)

        # Ignore samples that were labeled already.
        action_arr = action_arr[np.logical_not(labeled_mask)]
        # Keep sample indices parallel to action_arr
        sample_idxs = sample_idxs[np.logical_not(labeled_mask)]

        # Get the indices of the sample_num highest values.
        sorted_scores = np.argsort(action_arr)

        # get sample_num highest values. We use sample_idxs to keep track and return in terms of the original indices.
        return sample_idxs[sorted_scores[-sample_num:]]

    # JL - Nothing needed for this simple method.
    def update_on_new_state(self, new_state, new_state_labeled_mask, previous_state, previous_state_labeled_mask):
        pass

# Adish
# random

class RandomSamplingStrategy(ActiveLearningMethod):
    def __init__(self):
        super().__init__()

    def choose_n_samples(self, sample_num, state, labeled_mask):
        
        
        unlabeled_indices = np.where(labeled_mask == 0)[0]
        
        # Randomly select 'sample_num' indices
        if len(unlabeled_indices) > sample_num:
            selected_indices = np.random.choice(unlabeled_indices, size=sample_num, replace=False)
        else:
            selected_indices = unlabeled_indices

        return selected_indices

    def update_on_new_state(self, new_state, new_state_labeled_mask, previous_state, previous_state_labeled_mask):
       pass


# Adish
# Least confidence

class LeastConfidenceStrategy(ActiveLearningMethod):
    def __init__(self):
        super().__init__()

    def choose_n_samples(self, sample_num, state, labeled_mask):
        """
        Selects samples where the model has the least confidence in its predictions.
        """
        # Calculate the confidence 
        confidences = 1 - np.max(state, axis=1)
        # Finds indices of unlabeled samples
        unlabeled_indices = np.where(labeled_mask == 0)[0]
        
        # Filters confidence by unlabeled samples
        unlabeled_confidences = confidences[unlabeled_indices]
    
        # Gets the indices of samples sorted by confidence (least confident first)
        sorted_indices = np.argsort(unlabeled_confidences)
        
        # Select the top 'sample_num' indices
        selected_indices = sorted_indices[:sample_num]

        # Return these indices, reindexed to the original dataset
        return unlabeled_indices[selected_indices]

    def update_on_new_state(self, new_state, new_state_labeled_mask, previous_state, previous_state_labeled_mask):
        pass


# margin sampling

# FUSION - low priority

# For the below class, we ported and heavily modified Keras' example code from here: https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
# Much of the code in the class is courtesy of the original author: [amifunny](https://github.com/amifunny)
class DRLA(ActiveLearningMethod):

    # JL: TODO: Missing parts
    # 2) Some part of the flow needs to be reworked for getting actor action back to the critic, that's missing right now.

    def __init__(self, n_samples, k_classes, n_truth_labels):
        """
        Normal constructor for class. Subclass and pass in needed samples for the specific approach.
        """
        super().__init__()
        # parameters on the pool size:
        self.num_samples = n_samples
        self.num_classes = k_classes

        # This algorithm takes in all of the labels in the constructor at the beginning for efficiency.
        # HOWEVER, it doesn't use all of the labels when calculating reward.
        # Only the "revealed" truth labels are used at every update step. So it doesn't know the "future"!
        self.truth_labels = n_truth_labels

        # Define all hyperparameters for everything here.
        # JL - copying parameters from example for now.

        self.hidden_dense_units = 256
        self.critic_negative_slope = 0.3  # Q can't really be negative, but prevent the critic from completely losing

        self.noise_std_dev = 0.2
        self.ou_noise = DRLA.OUActionNoise(mean=np.zeros(1), std_deviation=float(self.noise_std_dev) * np.ones(1))

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.update_batch_size = 32

        self.critic_optimizer = keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(self.actor_lr)

        # JL - the example is suspiciously close to our paper for these parameters...
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        # End hyperparameters

        self.actor_model = self.build_actor()  # keras actor model
        self.critic_model = self.build_critic()  # keras critic model

        self.target_actor_model = self.build_actor()
        self.target_critic_model = self.build_critic()

        # Equal weights initially
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

        self.replay_buffer = DRLA.ReplayBuffer((n_samples, k_classes), n_samples)

    # JL - porting example code helper functions below ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """
    JL - modified functions that define critic and actor models.
    Our model architecture will be defined here, along with network parameters.
    
    """
    def build_actor(self):  # renamed from get_actor()

        inputs = layers.Input(shape=(self.num_samples, self.num_classes))
        flatten_input = layers.Flatten()(inputs)
        out = layers.Dense(self.hidden_dense_units, activation="relu")(flatten_input)
        out = layers.Dense(self.hidden_dense_units, activation="relu")(out)
        outputs = layers.Dense(self.num_samples, activation="sigmoid",)(out)

        # JL - the paper says 3 fully connected layers. Do we count the last sigmoid?

        model = keras.Model(inputs, outputs)
        return model

    def build_critic(self):  # renamed from get_critic()

        # JL - commented out original code + comments below. Our paper says the critic has the "same" architecture? Doesn't make sense...
        # State as input
        # state_input = layers.Input(shape=(self.num_samples, self.num_classes))
        # state_out = layers.Dense(16, activation="relu")(state_input)
        # state_out = layers.Dense(32, activation="relu")(state_out)
        #
        # # Action as input
        # action_input = layers.Input(shape=(num_actions,))
        # action_out = layers.Dense(32, activation="relu")(action_input)
        #
        # # Both are passed through seperate layer before concatenating
        # concat = layers.Concatenate()([state_out, action_out])
        #
        # out = layers.Dense(256, activation="relu")(concat)
        # out = layers.Dense(256, activation="relu")(out)
        # outputs = layers.Dense(1)(out)

        # # Outputs single value for give state-action
        # model = keras.Model([state_input, action_input], outputs)

        # JL - for now
        state_input = layers.Input(shape=(self.num_samples, self.num_classes), name="state_input")
        flat_state = layers.Flatten()(state_input)
        action_input = layers.Input(shape=(self.num_samples, 1), name="action_input")  # Whether a sample was labeled or not.
        flat_action = layers.Flatten()(action_input)
        concat_all = layers.Concatenate()([flat_state, flat_action])
        out = layers.Dense(self.hidden_dense_units, activation="relu")(concat_all)
        out = layers.Dense(self.hidden_dense_units, activation="relu")(out)
        out = layers.Dense(1, activation="linear", )(out)  # JL - in our problem, the Q value will never be below zero,
        # so we'll use an absolute value to keep it positive and avoid weirdness in case it happens to start negative!
        # Lambda layer: https://stackoverflow.com/questions/64856336/absolute-value-layer-in-cnn
        # JL - can review this in case critic doesn't fit at all!
        q_output = layers.Lambda(lambda x: tf.abs(x))(out)

        model = keras.Model(inputs=[state_input, action_input], outputs=q_output)

        return model

    @staticmethod
    def reward(y_pred, y_true):
        """
        Compute the reward on a set of predictions, as according to DRLA.
        This assumes everything is provided in one-hot encoding.
        :param y_pred: Classifier predicted values. The State! So the shape is (nsamples)x(kclasses).
        :param y_true: The ground-truth values. So the shape is (nsamples)x(kclasses).
        :return: The scalar reward value.
        """
        # Subtract the classifier prediction on the correct reward from the maximal confidence value.
        # Basically, if the correct value is the classifier's highest output, 0 reward.
        # But if the classifier's highest output is the wrong value, subtract the classifier's confidence on the correct value!

        # get ground truth indices.
        true_idxs = np.argmax(y_true, axis=1)

        # we average the rewards across all of the samples passed in!
        return np.mean(np.max(y_pred, axis=1) - y_pred[np.arange(y_pred.shape[0]), true_idxs])

    """
    To implement better exploration by the Actor network, we use noisy perturbations,
    specifically
    an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
    It samples noise from a correlated normal distribution.
    """

    # JL - TODO: Might need to tune this? Is this noise enough to affect our actions for exploration?
    class OUActionNoise:
        def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
            self.theta = theta
            self.mean = mean
            self.std_dev = std_deviation
            self.dt = dt
            self.x_initial = x_initial
            self.reset()

        def __call__(self):
            # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
            x = (
                    self.x_prev
                    + self.theta * (self.mean - self.x_prev) * self.dt
                    + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            )
            # Store x into x_prev
            # Makes next noise dependent on current one
            self.x_prev = x
            return x

        def reset(self):
            if self.x_initial is not None:
                self.x_prev = self.x_initial
            else:
                self.x_prev = np.zeros_like(self.mean)


    # Rough calculations on memory use of buffer; is it feasible to store in ram?
    # Assume max # of classes is 10
    # Assume max # of samples: 10000
    # Assume numpy arrays are float64, so 8 bytes per value.
    # To give our A/C system the best chance of fitting, we choose 1 sample at a time to label.
    # So for 10000 samples, this is 10000 states maximum.
    # Worst case for total number of. bytes: 8*10*10000^2 = 8*10^9
    # This is 8*10^9 bytes. So 7.45 GB.
    # For 2 classes at 10000 samples this is doable (~1.5 GB)
    # For 10 classes at 5000 samples it is much better as well (~1.86 GB)
    # JL NOTE/TODO: This should be implemented more efficiently for our case. It is functional for now, but wastes memory!
    # JL NOTE 2: Renaming this slightly. It is not just any general buffer...
    class ReplayBuffer:
        # JL - modifications to match our problem below.
        def __init__(self, state_shape_tuple, n_samples):
            # Number of "experiences" to store at max
            self.buffer_capacity = n_samples

            # Its tells us num of times record() was called.
            self.buffer_counter = 0

            # Instead of list of tuples as the exp.replay concept go
            # We use different np.arrays for each tuple element

            # JL - we have a unique shape for our state space, we concatenate the provided tuple.
            self.state_buffer = np.zeros((self.buffer_capacity,) + state_shape_tuple)
            self.action_buffer = np.zeros((self.buffer_capacity, n_samples))
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros((self.buffer_capacity,) + state_shape_tuple)  # JL - inefficient!!!

        # Takes (s,a,r,s') observation tuple as input
        def record(self, obs_tuple):
            # Set index to zero if buffer_capacity is exceeded,
            # replacing old records
            index = self.buffer_counter % self.buffer_capacity

            self.state_buffer[index] = obs_tuple[0]
            self.action_buffer[index] = obs_tuple[1]
            self.reward_buffer[index] = obs_tuple[2]
            self.next_state_buffer[index] = obs_tuple[3]

            self.buffer_counter += 1

        # JL - I reworked the following code, moved it outside of the ReplayBuffer class. The example's structure
        # is weird and doesn't make sense.
        def get_buffer_arrs(self, num_to_sample=None):
            """
            Get all 4 buffer components, in random order, to train actor & critic.
            :param num_to_sample: Number to choose if not None. Defaults to choosing all of it though.
            :return:
            """
            record_range = min(self.buffer_counter, self.buffer_capacity)

            # Randomly sample indices
            if num_to_sample is None:
                batch_indices = np.random.choice(record_range, record_range)
            else:
                batch_indices = np.random.choice(record_range, num_to_sample)

            return (self.state_buffer[batch_indices], self.action_buffer[batch_indices],
                    self.reward_buffer[batch_indices], self.next_state_buffer[batch_indices])


    # We compute the loss and update parameters
    def learn_actor_critic(self):
        # JL - batch size is currently the entire buffer!
        # May need to make into smaller batches for some training!
        state, action, reward, next_state = self.replay_buffer.get_buffer_arrs()

        batch_num = int(state.shape[0] / self.update_batch_size)

        if state.shape[0] % self.update_batch_size != 0:
            batch_num += 1

        # Shuffle every update!
        idxs = list(range(state.shape[0]))

        # shuffle
        random.shuffle(idxs)

        for i in range(batch_num):
            print("A/C batch update: " + str(i+1) + "/" + str(batch_num))
            batch_idxs = idxs[i*self.update_batch_size:(i+1)*self.update_batch_size]

            if len(batch_idxs) > 1:
                state_batch = state[batch_idxs]
                action_batch = action[batch_idxs]
                reward_batch = reward[batch_idxs]
                next_state_batch = next_state[batch_idxs]
            else:
                # only one element!
                state_batch = state
                action_batch = action
                reward_batch = reward
                next_state_batch = next_state

            # Convert to tensors
            state_batch = tf.convert_to_tensor(state_batch)
            action_batch = tf.convert_to_tensor(action_batch)
            reward_batch = tf.convert_to_tensor(reward_batch)
            reward_batch = tf.cast(reward_batch, dtype="float32")
            next_state_batch = tf.convert_to_tensor(next_state_batch)

            self.update(state_batch, action_batch, reward_batch, next_state_batch)

        # Apply update rule after going through the whole dataset.
        DRLA.update_target(self.target_actor_model, self.actor_model, self.tau)
        DRLA.update_target(self.target_critic_model, self.critic_model, self.tau)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_model(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic_model(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @staticmethod
    def update_target(target, original, tau):
        target_weights = target.get_weights()
        original_weights = original.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

        target.set_weights(target_weights)


    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """
    def policy(self, state, noise_object):
        # sampled_actions = keras.ops.squeeze(self.actor_model(state))
        sampled_actions = tf.squeeze(self.actor_model(np.expand_dims(state, axis=0)))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # JL - for our use case, we just return the values, then sort/decide!
        # # We make sure action is within bounds
        # legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return np.squeeze(sampled_actions)

    def choose_n_samples(self, sample_num, state, labeled_mask):
        """
        Based on the input state, choose samples to annotate.
        Let k = the number of classes. Let n = the total number of samples.
        :param sample_num: The number of samples to select for annotation. The annotator should select up to this many;
        it will select less if the unlabeled pool is smaller than this.
        :param state: The classifier prediction on all samples (labeled or unlabeled) in the pool.
        So a numpy array of shape nxk.
        :param labeled_mask: A mask that denotes which samples (True) have been "annotated" and which samples (False)
        have not been revealed to the classifier. Numpy array of size n.
        :return: Numpy array of indices (indexing into state), denoting which samples to reveal labels for.
        So a numpy array of size sample_num
        """

        # Run actor model on the state.
        action_arr = self.policy(state, self.ou_noise)

        # Remember the correct indices!
        sample_idxs = np.arange(0, state.shape[0], 1, dtype=np.int64)

        # Ignore samples that were labeled already.
        action_arr = action_arr[np.logical_not(labeled_mask)]
        # Keep sample indices parallel to action_arr
        sample_idxs = sample_idxs[np.logical_not(labeled_mask)]

        # Get the indices of the sample_num highest values.
        sorted_scores = np.argsort(action_arr)

        # get sample_num highest values. We use sample_idxs to keep track and return in terms of the original indices.
        return sample_idxs[sorted_scores[-sample_num:]]

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

        # Add sample to the replay buffer. All 4 parameters are used to add to the replay buffer

        # Loop through the entire replay buffer for one "epoch". Do the following for each sample (batch size of 1):
        # 1) Update the critic with one gradient update. Equation 6 from the paper
        # 2) Update the actor with one gradient update. Equation 4 from the paper
        # 3) Update target actor/critic with Equation 7 from the paper.

        print("Actor/Critic update start!")

        # If labels somehow decrease, there's a probem!
        assert np.count_nonzero(new_state_labeled_mask) > np.count_nonzero(previous_state_labeled_mask), "Somehow, samples got un-labeled!"

        # Calculate reward on all revealed samples.
        reward = DRLA.reward(new_state[new_state_labeled_mask], self.truth_labels[new_state_labeled_mask])

        # Calculate action be taking the difference between the masks
        action_mask = np.logical_xor(new_state_labeled_mask, previous_state_labeled_mask)

        # convert to 0.0/1.0 for critic NN input
        action = np.zeros((new_state.shape[0]))
        action[action_mask] = 1.0

        self.replay_buffer.record((previous_state, action, reward, new_state))

        self.learn_actor_critic()

        print("Actor/Critic update done!")

        # Done!