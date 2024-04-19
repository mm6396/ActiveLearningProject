# Jeremy Lim
# jlim@wpi.edu
# All of our active learning strategies are implemented here.

import abc

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

# zaher
# random

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
        sample_idxs = np.arange(0, sample_num.shape[0], 1, dtype=np.int64)

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
# Least confidence

# margin sampling

# FUSION - low priority

# For the below class, we ported and heavily modified Keras' example code from here: https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
# Much of the code is courtesy of the original author: [amifunny](https://github.com/amifunny)
class DRLA(ActiveLearningMethod):

    # JL: TODO: Missing parts
    # 1) Our reward function is unique, needs to replace MSE that's used here. (It's buried in it...)
    # 2) Some part of the flow needs to be reworked for getting actor action back to the critic, that's missing right now.
    # 3) Not sure if critic shape is correct or compatible yet.
    # 4) ActiveLearningMethod.update_on_new_state might not have the correct parameters. Need to update to make compatible with DRLA (Doesn't matter for other methods)

    def __init__(self, n_samples, k_classes):
        """
        Normal constructor for class. Subclass and pass in needed samples for the specific approach.
        """
        super().__init__()
        # parameters on the pool size:
        self.num_samples = n_samples
        self.num_classes = k_classes

        # Define all hyperparameters for everything here.
        # TODO: make parameters?
        # JL - copying parameters from example for now.

        self.noise_std_dev = 0.2
        self.ou_noise = DRLA.OUActionNoise(mean=np.zeros(1), std_deviation=float(self.noise_std_dev) * np.ones(1))

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

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

        self.replay_buffer = DRLA.ReplayBuffer((n_samples, k_classes), n_samples, batch_size=64)



    # JL - porting example code helper functions below ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """
    JL - modified functions that define critic and actor models.
    Our model architecture will be defined here, along with network parameters.
    
    """
    def build_actor(self):  # renamed from get_actor()

        inputs = layers.Input(shape=(self.num_samples, self.num_classes))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="sigmoid",)(out)

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

        # Exactly the same as actor? This is what the paper says - JL
        inputs = layers.Input(shape=(self.num_samples, self.num_classes))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="sigmoid", )(out)

        # JL - the paper says 3 fully connected layers. Do we count the last sigmoid?

        model = keras.Model(inputs, outputs)

        return model

    """
    To implement better exploration by the Actor network, we use noisy perturbations,
    specifically
    an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
    It samples noise from a correlated normal distribution.
    """

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
            self.state_buffer = np.zeros(self.buffer_capacity + state_shape_tuple)
            num_actions = None # JL - Rework
            self.action_buffer = np.zeros((self.buffer_capacity, num_actions))  # TODO: JL - What do we pass for this
            self.reward_buffer = np.zeros((self.buffer_capacity, 1))
            self.next_state_buffer = np.zeros(self.buffer_capacity + state_shape_tuple)  # JL - inefficient!!!

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

        state, action, reward, next_state = self.replay_buffer.get_buffer_arrs()

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(state)
        action_batch = keras.ops.convert_to_tensor(action)
        reward_batch = keras.ops.convert_to_tensor(reward)
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(next_state)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

        # Apply update rule
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
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

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
        sampled_actions = keras.ops.squeeze(self.actor_model(state))
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
        sample_idxs = np.arange(0, sample_num.shape[0], 1, dtype=np.int64)

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

        # Done. All internal state that needed to be updated is finished.
        #

        # TODO: JL - does our particular critic take a reward/action?
        action = None # JL - needs reworking!!!
        reward = None # JL - needs reworking!!!
        self.replay_buffer.record((previous_state, action, reward, new_state))

        self.learn_actor_critic()